/**
 * @file WeightPlan.cpp
 * @brief Implementation of declarative weight plans and frozen binding lookup.
 *
 * Device-qualified lookup deliberately scans the small graph-time binding
 * collection.  Graph construction is cold-path work, and this makes an
 * ambiguity a precise fatal diagnostic instead of an insertion-order-dependent
 * route to the wrong heterogeneous expert slice. Explicit static expert
 * projections derive original-axis slices from one immutable ownership value;
 * compact matrix views retain that same source/policy through preparation.
 * Static lowering preserves ordinary bindings in the same complete frozen set,
 * so graph preparation retains one original persistent owner and source lifetime.
 */

#include "WeightPlan.h"

#include "WeightLifecycleTrace.h"
#include "MoEExpertSourceView.h"
#include "tensors/TensorClasses.h"
#include "fort.hpp"

#include <sstream>
#include <utility>
#include <set>
#include <tuple>

namespace llaminar2
{
    namespace
    {
        /** @return Exact source coordinates, including original expert order and compactness. */
        bool sameSourceSlice(const WeightSliceSpec &a, const WeightSliceSpec &b)
        {
            const auto fields = [](const WeightSliceSpec &slice) {
                return std::tie(slice.source_rows, slice.source_cols, slice.row_start, slice.row_count,
                    slice.col_start, slice.col_count, slice.expert_start, slice.expert_count,
                    slice.expert_ids, slice.inner_is_presliced);
            };
            return fields(a) == fields(b);
        }

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

    namespace
    {
        /** @return Whether an owner may back a persistent prepared-weight set. */
        bool isPersistentWeightOwner(PhysicalMemoryOwner owner) noexcept
        {
            return owner == PhysicalMemoryOwner::PrimaryModelWeights ||
                   owner == PhysicalMemoryOwner::AdditionalModelWeights ||
                   owner == PhysicalMemoryOwner::RoutedExpertWeights;
        }
    } // namespace

    WeightPlan::WeightPlan(
        InferenceStrategy strategy,
        PhysicalMemoryOwner physical_memory_owner)
        : strategy_(std::move(strategy)),
          physical_memory_owner_(physical_memory_owner)
    {
        if (!isPersistentWeightOwner(physical_memory_owner_))
        {
            throw std::invalid_argument(
                "WeightPlan requires a persistent prepared-weight physical-memory owner");
        }
    }

    void WeightPlan::add(WeightRequirement requirement)
    {
        if (requirement.role == WeightRole::Other && !requirement.canonical_name.empty())
            requirement.role = inferWeightRole(requirement.canonical_name);
        if (requirement.layer < 0)
            requirement.layer = inferWeightLayer(requirement.canonical_name);
        if (requirement.expert < 0)
            requirement.expert = inferWeightExpert(requirement.canonical_name);
        if (requirement.projection_ownership)
        {
            const auto &ownership = *requirement.projection_ownership;
            const auto source_name = requirement.source_name.empty()
                ? requirement.canonical_name : requirement.source_name;
            if (ownership.hasMovableProjectionSet() || !isRoutedExpertRole(requirement.role) ||
                inferWeightRole(source_name) != requirement.role ||
                inferWeightRole(requirement.canonical_name) != requirement.role ||
                inferWeightLayer(source_name) != requirement.layer ||
                inferWeightLayer(requirement.canonical_name) != requirement.layer ||
                requirement.tp_rank_or_device_index != ownership.participant() ||
                !requirement.lookup_device.value_or(requirement.target_device).is_cpu() ||
                requirement.derivation != WeightDerivationKind::ExpertSlice ||
                requirement.expert >= 0 || !requirement.bypass_tensor_parallel ||
                (requirement.expected_prepared_kind != PreparedWeightKind::None &&
                 requirement.expected_prepared_kind != PreparedWeightKind::MoeExpertSlab) ||
                (requirement.overlay_participant_index >= 0 &&
                 requirement.overlay_participant_index != ownership.participant()))
                throw std::invalid_argument("Static expert projection requires its exact original role/participant source request");
            const auto projection = ownership.projection(requirement.role);
            WeightSliceSpec original;
            // The loader consumes original GGUF [K,N,E], not GEMM [N,K].
            original.source_rows = static_cast<size_t>(projection.source_columns);
            original.source_cols = static_cast<size_t>(projection.source_rows);
            original.row_start = static_cast<size_t>(projection.first_column);
            original.row_count = static_cast<size_t>(projection.columns);
            original.col_start = static_cast<size_t>(projection.first_row);
            original.col_count = static_cast<size_t>(projection.rows);
            original.expert_count = static_cast<size_t>(ownership.geometry().experts);
            original.expert_ids.reserve(original.expert_count);
            for (int expert = 0; expert < ownership.geometry().experts; ++expert)
                original.expert_ids.push_back(expert);
            // An already normalized requirement may be composed into another
            // plan, but contradictory caller coordinates are never substituted.
            if (!sameSourceSlice(requirement.slice, WeightSliceSpec{}) &&
                !sameSourceSlice(requirement.slice, original))
                throw std::invalid_argument("Static expert source coordinates disagree with immutable projection ownership");
            requirement.slice = std::move(original);
            // This requests original source bytes. Only materialization may
            // publish inner_is_presliced after the exact compact copy.
        }
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

    FrozenModelWeightSet::FrozenModelWeightSet(
        InferenceStrategy strategy,
        std::vector<WeightBinding> bindings,
        PhysicalMemoryOwner physical_memory_owner)
        : strategy_(std::move(strategy)),
          physical_memory_owner_(physical_memory_owner),
          bindings_(std::move(bindings))
    {
        if (!isPersistentWeightOwner(physical_memory_owner_))
        {
            throw std::invalid_argument(
                "FrozenModelWeightSet requires a persistent prepared-weight physical-memory owner");
        }
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

    const WeightBinding *FrozenModelWeightSet::optionalLayerForDevice(
        int layer_idx,
        const std::string &suffix,
        DeviceId device) const
    {
        const WeightBinding *match = nullptr;
        for (const auto &binding : bindings_)
        {
            if (binding.identity.layer != layer_idx ||
                suffixForLayerWeight(binding.identity.canonical_name) != suffix)
            {
                continue;
            }

            const bool resident_on_device =
                binding.residency.home_device == device ||
                (binding.residency.resident_device &&
                 *binding.residency.resident_device == device);
            if (!resident_on_device)
                continue;

            if (match)
            {
                throw std::runtime_error(
                    "Ambiguous device-qualified layer binding for layer=" +
                    std::to_string(layer_idx) + " suffix=" + suffix +
                    " device=" + device.to_string() +
                    "; every graph must have one authoritative binding");
            }
            match = &binding;
        }
        return match;
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

    /** @copydoc FrozenModelWeightSet::projectedExpertMatrices */
    FrozenModelWeightSet FrozenModelWeightSet::projectedExpertMatrices() const
    {
        validateForGraph();
        if (bindings_.empty())
            throw std::invalid_argument("Static expert projection cannot publish an empty source set");
        ModelWeightSetBuilder builder(strategy_);
        std::set<uint64_t> identifiers;
        std::set<uint64_t> original_identifiers;
        bool has_static_parent = false;
        // Reserve preserved identities before expanding any parent: order must
        // not decide whether a generated expert key aliases a dense binding.
        for (const auto &binding : bindings_)
        {
            if (!binding.binding_id || !original_identifiers.insert(binding.binding_id).second)
                throw std::invalid_argument("Static expert lowering requires distinct original binding identities");
            if (binding.projection_ownership)
                has_static_parent = true;
            else
                identifiers.insert(binding.binding_id);
        }
        if (!has_static_parent)
            throw std::invalid_argument("Static expert lowering requires an original marked source parent");
        for (const auto &parent : bindings_)
        {
            if (!parent.projection_ownership)
            {
                // This is the original frozen binding, not a reconstructed
                // dense requirement or a second prepared-store publication.
                builder.addBinding(parent);
                continue;
            }
            if (parent.projection_ownership->hasMovableProjectionSet() ||
                !parent.tensor || parent.tensor_owner.get() != parent.tensor ||
                parent.tensor->shape().size() != 3 || !isRoutedExpertRole(parent.identity.role) ||
                parent.identity.derivation != WeightDerivationKind::ExpertSlice || !parent.slice.inner_is_presliced)
                throw std::invalid_argument("Projected expert matrices require owned compact static source bindings");
            if (parent.tensor->is_raw_data_released() || !parent.tensor->raw_data())
                throw std::runtime_error("Static expert source retired before its matrix bindings were created");
            const auto &ownership = *parent.projection_ownership;
            const MoEExpertSourceView source(parent.tensor->shape(), parent.slice);
            if (source.experts().size() != static_cast<size_t>(ownership.geometry().experts))
                throw std::invalid_argument("Static expert source must retain the complete original expert inventory");
            for (int expert = 0; expert < ownership.geometry().experts; ++expert)
            {
                if (source.experts()[static_cast<size_t>(expert)] != expert)
                    throw std::invalid_argument("Static expert source does not preserve original ordinal IDs");
                const auto matrix = source.matrix(expert, parent.identity.role, &ownership);
                auto view = parent.tensor->create_view({matrix.rows, matrix.columns}, matrix.element_offset);
                if (!view || view->native_type() != parent.tensor->native_type() ||
                    parent.tensor->size_bytes() % static_cast<size_t>(ownership.geometry().experts) ||
                    view->size_bytes() != parent.tensor->size_bytes() / static_cast<size_t>(ownership.geometry().experts))
                    throw std::runtime_error("Static expert matrix view lost its native source representation");
                WeightBinding binding = parent;
                binding.identity.canonical_name = parent.identity.canonical_name + "@expert" + std::to_string(expert);
                // Parent identity already binds full source, participant and all
                // F intervals. Domain-separated byte mixing adds the exact E slot.
                uint64_t identifier = parent.binding_id ^ 1469598103934665603ull;
                for (const unsigned char byte : binding.identity.canonical_name)
                {
                    identifier ^= byte;
                    identifier *= 1099511628211ull;
                }
                if (!identifier) identifier = 1;
                if (!identifiers.insert(identifier).second)
                    throw std::logic_error("Static expert matrix binding identity collision");
                binding.binding_id = identifier;
                binding.identity.logical_id = stableWeightLogicalId(binding.identity.canonical_name);
                binding.identity.source_instance_id = parent.identity.instance_id;
                binding.identity.instance_id = identifier;
                binding.identity.expert = expert;
                binding.slice = matrix.source_identity;
                binding.projection_source_owner = parent.tensor_owner; // Same actual source, not a copied inventory.
                binding.tensor_owner = std::move(view); // Existing view retains its original shared source root.
                binding.tensor = binding.tensor_owner.get();
                const auto device = binding.residency.resident_device.value_or(binding.residency.home_device);
                const auto kind = device.is_cuda() ? PreparedWeightKind::CudaInt8PackedGemm
                    : device.is_rocm() ? PreparedWeightKind::RocmInt8PackedGemm : PreparedWeightKind::CpuPackedGemm;
                binding.prepared = PreparedWeightRef{strategy_.model_id, identifier, kind, device};
                builder.addBinding(std::move(binding));
            }
        }
        FrozenModelWeightSet result(strategy_, builder.freezeBindings(), physical_memory_owner_);
        result.validateForGraph();
        return result;
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
