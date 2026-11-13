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
 * on insertion order. One immutable candidate index accelerates these queries
 * without copying residency or changing original binding addresses.
 */

#pragma once

#include "WeightIdentity.h"
#include "tensors/NativeWeightSource.h"
#include "execution/moe/MoEExpertProjectionOwnership.h"
#include "WeightLifecycleTrace.h"
#include "kernels/FloatingActivationOperandPrecision.h"
#include "planning/PhysicalMemoryBOM.h"

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

    class CPUExpertShardSourceSelection;

    enum class PreparedWeightKind
    {
        None,
        CpuPackedGemm,
        CudaInt8PackedGemm,
        RocmInt8PackedGemm,
        PreparedEmbedding,
        MoeExpertSlab,
        CPUResidentTensor, ///< Committed final CPU tensor owner; never a GEMM engine or raw-source cache.
    };

    struct PreparedWeightRef
    {
        ModelContextId model_id;
        uint64_t binding_id = 0;
        PreparedWeightKind kind = PreparedWeightKind::None;
        DeviceId device = DeviceId::cpu();
    };

    /** @brief Frozen source, arithmetic and ownership consumed by one graph binding. */
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
        FloatingActivationOperandPrecision floating_operand_precision =
            FloatingActivationOperandPrecision::Native; ///< Sealed before preparation; part of reuse identity.
        std::shared_ptr<const CPUExpertShardSourceSelection> cpu_joint_source; ///< Closed original CPU joint-fragment purpose; absence preserves existing full-E GPU sources.
        std::optional<NativeWeightSourceProjection> native_source; ///< Original native source facts retained after committed final preparation; grants no ownership.
        std::optional<MoEExpertProjectionOwnership> projection_ownership; ///< Original static source partition; no topology permission.
        std::shared_ptr<TensorBase> projection_source_owner = {}; ///< Original compact3D source root for exact native staging/coalescing.
    };

    /** @brief Preserve the existing original-parent/expert matrix identity independent of staging windows.
     * @param parent Original complete requirement binding identity.
     * @param name Original canonical parent name.
     * @param expert Original logical expert ID.
     * @return Existing domain-separated matrix binding identity.
     */
    uint64_t staticProjectedExpertBindingId(uint64_t parent,const std::string &name,int expert);

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
        std::optional<NativeWeightSourceAssembly> native_source_assembly; ///< Ordered original native stripes for a dense final consumer, never a compact source parent.
        std::shared_ptr<const CPUExpertShardSourceSelection> cpu_joint_source; ///< Closed original CPU joint-fragment purpose; absence preserves existing full-E GPU sources.
        std::optional<MoEExpertProjectionOwnership> projection_ownership; ///< When present, the plan derives the sole original-axis slice.
    };

    /**
     * @brief Project original source coordinates for complete tensors and selected expert matrices.
     * @param requirement Original plan-owned axes and selected logical IDs.
     * @param source Original loader geometry in GGUF K/N/outer order.
     * @return Passive exact source rectangles; this does not allocate or grant source authority.
     * @throws std::invalid_argument For foreign bounds, unordered IDs or split native groups.
     */
    std::vector<NativeWeightSourceProjection> projectNativeWeightSource(
        const WeightRequirement &requirement, const NativeWeightSourceGeometry &source);

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
        /**
         * @brief Create an empty plan with its execution and memory authority.
         *
         * The owner is a property of the complete frozen set rather than a
         * call-site hint supplied during allocation. This prevents a mirrored
         * decode/MTP sidecar from being admitted as
         * `AdditionalModelWeights` and later materialized against the primary
         * owner line merely because both sets use the same loader pipeline.
         *
         * @param strategy Execution strategy served by the plan.
         * @param physical_memory_owner Exact persistent-weight BOM owner.
         * @throws std::invalid_argument for a non-weight memory owner.
         */
        explicit WeightPlan(
            InferenceStrategy strategy = {},
            PhysicalMemoryOwner physical_memory_owner =
                PhysicalMemoryOwner::PrimaryModelWeights);

        /** @return Immutable execution strategy carried by this plan. */
        const InferenceStrategy &strategy() const { return strategy_; }
        /** @return Exact physical-memory owner carried into materialization. */
        PhysicalMemoryOwner physicalMemoryOwner() const noexcept
        {
            return physical_memory_owner_;
        }
        /** @return Ordered requirements that will be materialized. */
        const std::vector<WeightRequirement> &requirements() const { return requirements_; }
        /**
         * @brief Normalize and append one declarative tensor requirement.
         * @param requirement Original source/target intent. An explicit static projection
         *        derives its original-axis all-expert slice from the retained ownership.
         * @throws std::invalid_argument for a foreign role/source/participant, unsupported
         *         parent preparation intent, or contradictory caller source coordinates.
         */
        void add(WeightRequirement requirement);
        /** @return Number of declarative requirements. */
        size_t size() const { return requirements_.size(); }
        /** @return Whether the plan contains no requirements. */
        bool empty() const { return requirements_.empty(); }
        /** @return A human-readable audit table for diagnostics and CSV evidence. */
        std::string renderAuditTable() const;

    private:
        InferenceStrategy strategy_;
        PhysicalMemoryOwner physical_memory_owner_ =
            PhysicalMemoryOwner::PrimaryModelWeights;
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
        FrozenModelWeightSet(
            InferenceStrategy strategy,
            std::vector<WeightBinding> bindings,
            PhysicalMemoryOwner physical_memory_owner =
                PhysicalMemoryOwner::PrimaryModelWeights);

        /** @return Execution strategy used to materialize these bindings. */
        const InferenceStrategy &strategy() const { return strategy_; }
        /** @return Exact persistent-weight owner certified by the source plan. */
        PhysicalMemoryOwner physicalMemoryOwner() const noexcept
        {
            return physical_memory_owner_;
        }
        /** @return All immutable bindings in deterministic plan order. */
        const std::vector<WeightBinding> &bindings() const { return bindings_; }
        /** @brief Return a required model-global binding or throw when absent. */
        const WeightBinding &global(const std::string &canonical_name) const;
        /** @brief Return a required layer binding or throw when absent. */
        const WeightBinding &layer(int layer_idx, const std::string &suffix) const;
        /**
         * @brief Return the last in-plan unqualified layer binding.
         * @param layer_idx Original nonnegative model layer.
         * @param suffix Canonical layer-relative source name.
         * @return Original binding address, or nullptr for an absent layer/name.
         */
        const WeightBinding *optionalLayer(int layer_idx, const std::string &suffix) const;
        /**
         * @brief Return the sole layer binding resident on @p device.
         *
         * A null result means that the device does not own the requested
         * binding.  More than one matching binding is an invalid graph
         * authority and throws instead of selecting by insertion order. The
         * immutable name index narrows the candidates; original bindings remain
         * the sole residency authority, including home/resident aliases.
         * @param layer_idx Original binding layer coordinate.
         * @param suffix Canonical layer-relative source name (global name for a negative coordinate).
         * @param device Exact backend and ordinal, never an ordinal-only alias.
         * @return Sole original matching binding, or nullptr when absent.
         * @throws std::runtime_error If more than one original binding matches.
         */
        const WeightBinding *optionalLayerForDevice(
            int layer_idx,
            const std::string &suffix,
            DeviceId device) const;
        /** @brief Return all bindings whose home or resident device is @p device. */
        std::vector<const WeightBinding *> forDevice(DeviceId device) const;
        /** @brief Verify immutable IDs and prepared-handle identity invariants. */
        void validateForGraph() const;
        /**
         * @brief Bind each original expert's compact 2D source without copying payloads.
         * @return One complete frozen model: ordinary bindings remain unchanged,
         *         while marked parents become participant-local GEMM views retaining their original 3D owner,
         *         full source partition, per-expert identity and persistent memory owner.
         * @throws std::exception for absent/ambiguous source geometry or unsupported layouts.
         *
         * Preserved ordinary IDs are reserved before expansion; generated keys
         * cannot collide with them regardless of input order. At least one
         * compact marked parent is required; an already lowered set is rejected.
         * Startup-only source bindings, not graph/registry/topology admission.
         * Source-native GPU packing additionally requires the original declared
         * participant scope. Graph/topology integration remains separate.
         */
        FrozenModelWeightSet projectedExpertMatrices() const;
        /** @return A human-readable audit table for diagnostics and evidence. */
        std::string renderAuditTable() const;

    private:
        /**
         * @brief Index one original binding without copying its ownership.
         * @param index Stable position in the immutable bindings_ sequence.
         * @param binding Binding at that position; source order defines legacy lookup.
         */
        void indexBinding(size_t index, const WeightBinding &binding);

        InferenceStrategy strategy_;
        PhysicalMemoryOwner physical_memory_owner_ =
            PhysicalMemoryOwner::PrimaryModelWeights;
        std::vector<WeightBinding> bindings_;
        std::unordered_map<std::string, size_t> global_index_;
        // One name index retains all original candidates. Qualified lookup
        // checks residency here; unqualified lookup explicitly chooses the
        // greatest sequence index, independent of unordered iteration order.
        // Negative coordinates preserve qualified global-name queries only.
        std::unordered_multimap<std::string, size_t> layer_index_;
    };

    std::string toString(PreparedWeightKind kind);
}
