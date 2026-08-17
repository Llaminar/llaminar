/**
 * @file WeightPlan.h
 * @brief Declarative model-weight planning and immutable binding lookup.
 *
 * A @ref WeightPlan describes every source tensor, derivation, target device,
 * and prepared representation before any graph is built.  A
 * @ref FrozenModelWeightSet is the graph-time authority created from that
 * plan.  In particular, heterogeneous ExpertOverlay plans may contain more
 * than one binding for a routed-expert parent; callers that build a graph for
 * a concrete device must use the device-qualified lookup rather than relying
 * on insertion order.
 */

#pragma once

#include "WeightIdentity.h"
#include "WeightLifecycleTrace.h"

#include <functional>
#include <memory>
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
        std::shared_ptr<TensorBase> tensor_owner;
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
        WeightResidencyCategory residency_category = WeightResidencyCategory::Unspecified;
        std::string overlay_domain;
        int overlay_participant_index = -1;
        int overlay_participant_world_rank = -1;
        DeviceId target_device = DeviceId::cpu();
        std::optional<DeviceId> lookup_device;
        bool bypass_tensor_parallel = false;
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

    /** @brief Immutable-intent list used to materialize one model weight view. */
    class WeightPlan
    {
    public:
        /** @brief Create an empty plan with the execution strategy it serves. */
        explicit WeightPlan(InferenceStrategy strategy = {});

        /** @return Immutable execution strategy carried by this plan. */
        const InferenceStrategy &strategy() const { return strategy_; }
        /** @return Ordered requirements that will be materialized. */
        const std::vector<WeightRequirement> &requirements() const { return requirements_; }
        /** @brief Normalize and append one declarative tensor requirement. */
        void add(WeightRequirement requirement);
        /** @return Number of declarative requirements. */
        size_t size() const { return requirements_.size(); }
        /** @return Whether the plan contains no requirements. */
        bool empty() const { return requirements_.empty(); }
        /** @return A human-readable audit table for diagnostics and CSV evidence. */
        std::string renderAuditTable() const;

    private:
        InferenceStrategy strategy_;
        std::vector<WeightRequirement> requirements_;
    };

    /** @brief Mutable construction helper that assigns immutable binding IDs. */
    class ModelWeightSetBuilder
    {
    public:
        /** @brief Start a binding builder for one execution strategy. */
        explicit ModelWeightSetBuilder(InferenceStrategy strategy = {});

        /** @brief Add a binding and assign missing binding/instance identifiers. */
        WeightBinding &addBinding(WeightBinding binding);
        /** @brief Mark all bindings immutable and transfer their ownership. */
        std::vector<WeightBinding> freezeBindings();
        /** @return Strategy inherited by bindings created through this builder. */
        const InferenceStrategy &strategy() const { return strategy_; }

    private:
        InferenceStrategy strategy_;
        uint64_t next_binding_id_ = 1;
        std::vector<WeightBinding> bindings_;
    };

    /**
     * @brief Immutable graph-time lookup authority for materialized weights.
     *
     * The unqualified layer lookup remains available for single-binding plans.
     * Heterogeneous plans must select bindings with @ref optionalLayerForDevice
     * so a CPU tier's exact expert slice can never shadow a CUDA or ROCm tier
     * merely because it was appended later to the plan.
     */
    class FrozenModelWeightSet
    {
    public:
        /**
         * @brief Construct and index immutable materialized bindings.
         *
         * Call @ref validateForGraph before execution to verify that every
         * binding came from @ref ModelWeightSetBuilder::freezeBindings.
         */
        FrozenModelWeightSet(InferenceStrategy strategy, std::vector<WeightBinding> bindings);

        /** @return Execution strategy used to materialize these bindings. */
        const InferenceStrategy &strategy() const { return strategy_; }
        /** @return All immutable bindings in deterministic plan order. */
        const std::vector<WeightBinding> &bindings() const { return bindings_; }
        /** @brief Return a required model-global binding or throw when absent. */
        const WeightBinding &global(const std::string &canonical_name) const;
        /** @brief Return a required layer binding or throw when absent. */
        const WeightBinding &layer(int layer_idx, const std::string &suffix) const;
        /** @brief Return an unqualified layer binding, or nullptr when absent. */
        const WeightBinding *optionalLayer(int layer_idx, const std::string &suffix) const;
        /**
         * @brief Return the sole layer binding resident on @p device.
         *
         * A null result means that the device does not own the requested
         * binding.  More than one matching binding is an invalid graph
         * authority and throws instead of selecting by insertion order.
         */
        const WeightBinding *optionalLayerForDevice(
            int layer_idx,
            const std::string &suffix,
            DeviceId device) const;
        /** @brief Return all bindings whose home or resident device is @p device. */
        std::vector<const WeightBinding *> forDevice(DeviceId device) const;
        /** @brief Verify immutable IDs and prepared-handle identity invariants. */
        void validateForGraph() const;
        /** @return A human-readable audit table for diagnostics and evidence. */
        std::string renderAuditTable() const;

    private:
        /** @brief Populate the legacy unqualified lookup indexes for one binding. */
        void indexBinding(size_t index, const WeightBinding &binding);

        InferenceStrategy strategy_;
        std::vector<WeightBinding> bindings_;
        std::unordered_map<std::string, size_t> global_index_;
        std::unordered_map<std::string, size_t> layer_index_;
    };

    std::string toString(PreparedWeightKind kind);
}
