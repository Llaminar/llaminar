/**
 * @file PreparedWeightPhysicalUses.cpp
 * @brief Compile actual original-source alias edges before canonical weight pricing.
 *
 * A complete ordinary source matrix may reuse the primary prepared root under
 * the same runtime identity predicate. No source-name discount or estimated
 * residency credit exists here. Unknown, sliced and projected source facts
 * retain independent creators until their original owner can be proved.
 */
#include "planning/PreparedWeightPhysicalUses.h"
#include "planning/ModelMemoryProfile.h"
#include "planning/WeightShardGeometry.h"
#include "loaders/PreparedWeightRepresentationContract.h"
#include "execution/local_execution/graph/SchemaFactoryRegistry.h"
#include "utils/DebugEnv.h"
#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace llaminar2
{
    namespace
    {
        /** @return Actual inventory scalar/codebook enum, or no provable prepared identity. */
        std::optional<TensorType> inventoryType(std::string_view name)
        {
            if (name == "F32") return TensorType::FP32;
            if (name == "F16") return TensorType::FP16;
            for (int value = 0; value <= static_cast<int>(TensorType::AQ8); ++value)
                if (name == tensorTypeName(static_cast<TensorType>(value))) return static_cast<TensorType>(value);
            return std::nullopt;
        }
    }
    PreparedWeightPhysicalUses::PreparedWeightPhysicalUses(const ModelMemoryProfile &profile, DeviceId device,
        WeightComponentScope components)
        : profile_(profile), device_(device), components_(components)
    {
        for (const auto &source : profile.tensors) originals_.insert(&source);
        if (!weightScopeOwnsTerminal(components_))
        {
            const auto schema = SchemaFactoryRegistry::getFactory(profile.architecture);
            for (const auto *name : schema->terminalInputWeightNames()) terminal_sources_.insert(name);
        }
    }

    void PreparedWeightPhysicalUses::requireScope(const ModelMemoryProfile &profile, DeviceId device,
        WeightComponentScope components) const
    {
        if (&profile != &profile_ || device != device_ || components != components_)
            throw std::invalid_argument("Prepared weight physical-use compilation has a foreign model/device/component scope");
    }

    std::optional<PreparedWeightIdentityContract> PreparedWeightPhysicalUses::identity(
        const TensorSizeInfo &tensor, const WeightShardGeometry &geometry, WeightSourceStorageView source_view) const
    {
        // Source identity is the original inventory object in this compiler's
        // model epoch. A copied entry, even with the same name, is not authority.
        const bool original = originals_.find(&tensor) != originals_.end();
        if (!original || source_view != WeightSourceStorageView::WholeOriginal || isRoutedExpertRole(inferWeightRole(tensor.name)) || tensor.K == 0 ||
            !geometry.matrix() || geometry.matrix()->instances != 1 || geometry.elements() != tensor.elements ||
            geometry.matrix()->columns != tensor.K || tensor.elements % tensor.K ||
            geometry.matrix()->rows != tensor.elements / tensor.K) return std::nullopt;
        // Factory materialization publishes terminal inputs only on an actual
        // terminal component owner. A legacy byte declaration alone cannot
        // authenticate an otherwise absent primary creator on an earlier stage.
        if (!weightScopeOwnsTerminal(components_) && terminal_sources_.find(tensor.name) != terminal_sources_.end())
            return std::nullopt;
        const auto role = inferWeightRole(tensor.name);
        const auto classification = PreparedWeightIdentityContract::classifyOriginalMatrix(role, device_);
        if (classification == PreparedGemmClassification::Rejected) return std::nullopt;
        if (classification == PreparedGemmClassification::RequiresSchema)
        {
            // Exactly the existing WeightManager schema fallback. No schema
            // means no alias proof; it does not select another execution mode.
            try
            {
                const auto schema = SchemaFactoryRegistry::getWeightShardingConfig(profile_.architecture);
                if (schema.isNonGemmWeight(tensor.name)) return std::nullopt;
            }
            catch (const std::exception &) { return std::nullopt; }
        }
        auto type = inventoryType(tensor.quant_type);
        if (!type) return std::nullopt;
        const auto &policy = debugEnv().gemm;
        const auto representation = PreparedWeightRepresentationContract::resolve(role, *type, tensor.name,
            profile_.architecture, policy.gdn_projection_arithmetic, policy.moe_router_arithmetic);
        if (representation == ModelPreparedWeightRepresentation::FP16) type = TensorType::FP16;
        if (representation == ModelPreparedWeightRepresentation::FP32) type = TensorType::FP32;
        return PreparedWeightIdentityContract::completeOriginalMatrix(geometry.matrix()->rows,
            geometry.matrix()->columns, *type, PreparedWeightRepresentationContract::operandPrecision(role,
                tensor.name, profile_.architecture, policy.gdn_projection_arithmetic, policy.moe_router_arithmetic));
    }
    PreparedWeightPhysicalUse PreparedWeightPhysicalUses::record(const ModelMemoryProfile &profile, DeviceId device,
        const TensorSizeInfo &tensor, const WeightShardGeometry &geometry, WeightSourceStorageView source_view,
        PreparedWeightUseIntent intent)
    {
        if (intent != PreparedWeightUseIntent::PublishCreator && intent != PreparedWeightUseIntent::ReuseExistingOrCreate)
            throw std::invalid_argument("Prepared weight physical-use compilation has an invalid use intent");
        if (&profile != &profile_ || device != device_)
            throw std::invalid_argument("Prepared weight physical-use compilation has a foreign model/device scope");
        auto view = identity(tensor, geometry, source_view);
        if (intent == PreparedWeightUseIntent::ReuseExistingOrCreate && view)
            for (size_t root : roots_by_source_[&tensor])
                if (roots_[root].identity &&
                    roots_[root].identity->compatible(*view))
                {
                    edges_.push_back(PreparedWeightPhysicalAlias{root});
                    return edges_.back();
                }
        const auto *original = originals_.find(&tensor) != originals_.end() ? &tensor : nullptr;
        roots_.push_back({original, std::move(view)});
        if (roots_.back().identity) roots_by_source_[&tensor].push_back(roots_.size() - 1);
        edges_.push_back(PreparedWeightPhysicalCreator{roots_.size() - 1});
        return edges_.back();
    }
}
