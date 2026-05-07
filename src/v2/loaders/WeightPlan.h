#pragma once

#include "WeightIdentity.h"
#include "WeightLifecycleTrace.h"

#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{
    class TensorBase;
    struct GraphConfig;

    enum class PreparedWeightKind
    {
        None,
        CpuPackedGemm,
        CudaInt8PackedGemm,
        RocmInt8PackedGemm,
        PreparedEmbedding,
        MoeExpertSlab,
    };

    struct PreparedWeightRef
    {
        ModelContextId model_id;
        uint64_t binding_id = 0;
        PreparedWeightKind kind = PreparedWeightKind::None;
        DeviceId device = DeviceId::cpu();
    };

    struct WeightBinding
    {
        uint64_t binding_id = 0;
        WeightIdentity identity;
        WeightSliceSpec slice;
        WeightResidency residency;
        TensorBase *tensor = nullptr;
        std::optional<PreparedWeightRef> prepared;
        bool immutable = false;
    };

    struct WeightRequirement
    {
        std::string canonical_name;
        std::string source_name;
        bool required = true;
        WeightRole role = WeightRole::Other;
        WeightDerivationKind derivation = WeightDerivationKind::Source;
        int layer = -1;
        int expert = -1;
        int pp_stage = -1;
        int tp_domain = -1;
        int tp_rank_or_device_index = 0;
        DeviceId target_device = DeviceId::cpu();
        std::optional<DeviceId> lookup_device;
        WeightHostPolicy host_policy = WeightHostPolicy::RequiredUntilGraphMaterialized;
        PreparedWeightKind expected_prepared_kind = PreparedWeightKind::None;
        WeightSliceSpec slice;
    };

    struct InferenceStrategy
    {
        WeightInferenceMode mode = WeightInferenceMode::Unknown;
        ModelContextId model_id;
        int pp_stages = 1;
        int tp_degree = 1;
        std::vector<DeviceId> devices;
    };

    class WeightPlan
    {
    public:
        explicit WeightPlan(InferenceStrategy strategy = {});

        const InferenceStrategy &strategy() const { return strategy_; }
        const std::vector<WeightRequirement> &requirements() const { return requirements_; }
        void add(WeightRequirement requirement);
        size_t size() const { return requirements_.size(); }
        bool empty() const { return requirements_.empty(); }
        std::string renderAuditTable() const;

    private:
        InferenceStrategy strategy_;
        std::vector<WeightRequirement> requirements_;
    };

    class ModelWeightSetBuilder
    {
    public:
        explicit ModelWeightSetBuilder(InferenceStrategy strategy = {});

        WeightBinding &addBinding(WeightBinding binding);
        std::vector<WeightBinding> freezeBindings();
        const InferenceStrategy &strategy() const { return strategy_; }

    private:
        InferenceStrategy strategy_;
        uint64_t next_binding_id_ = 1;
        std::vector<WeightBinding> bindings_;
    };

    class FrozenModelWeightSet
    {
    public:
        /// Bindings are logically immutable after construction. Call validateForGraph()
        /// before execution to enforce that every binding came from ModelWeightSetBuilder::freezeBindings().
        FrozenModelWeightSet(InferenceStrategy strategy, std::vector<WeightBinding> bindings);

        const InferenceStrategy &strategy() const { return strategy_; }
        const std::vector<WeightBinding> &bindings() const { return bindings_; }
        const WeightBinding &global(const std::string &canonical_name) const;
        const WeightBinding &layer(int layer_idx, const std::string &suffix) const;
        const WeightBinding *optionalLayer(int layer_idx, const std::string &suffix) const;
        std::vector<const WeightBinding *> forDevice(DeviceId device) const;
        void validateForGraph() const;
        std::string renderAuditTable() const;

    private:
        void indexBinding(size_t index, const WeightBinding &binding);

        InferenceStrategy strategy_;
        std::vector<WeightBinding> bindings_;
        std::unordered_map<std::string, size_t> global_index_;
        std::unordered_map<std::string, size_t> layer_index_;
    };

    std::string toString(PreparedWeightKind kind);
}
