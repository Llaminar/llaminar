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

    enum class WeightResidencyCategory
    {
        Unspecified,
        RootNonExpert,
        SharedExpert,
        AcceleratorRoutedExpert,
        CpuFallbackExpert,
        WorkerFallbackExpert,
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
        WeightResidencyCategory residency_category = WeightResidencyCategory::Unspecified;
        std::string overlay_domain;
        int overlay_participant_index = -1;
        int overlay_participant_world_rank = -1;

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

    // =========================================================================
    // Phase 9: Model-level weight lifecycle state machine
    // =========================================================================

    /**
     * @brief Global lifecycle state for all model weights within a context
     *
     * Represents the progression of model weights from loading through to
     * frozen execution state. Transitions are one-directional (monotonically
     * increasing). The state applies to the model context as a whole, not to
     * individual weights (individual per-weight readiness uses WeightPrepState).
     *
     * State transitions:
     *   Planned → SourceLoaded → DerivedMaterialized → DevicePrepared
     *            → GraphMaterialized → Frozen → HostReleased
     *
     * Each state implies all previous states are complete. For example,
     * DevicePrepared means all sources are loaded and all derived tensors
     * (slices, clones, TP shards) are materialized.
     */
    enum class WeightLifecycleState
    {
        /// Weights planned but not yet loaded from source (GGUF)
        Planned = 0,

        /// All source tensors loaded from model file into host memory
        SourceLoaded,

        /// All derived tensors (TP slices, PP layers, expert views, clones) materialized
        DerivedMaterialized,

        /// All GEMM/embedding weights prepared for their target devices
        /// (CPU packed, CUDA/ROCm uploaded, prepared handles created)
        DevicePrepared,

        /// Graph construction complete — all weight bindings resolved and stages
        /// hold frozen references to prepared weights
        GraphMaterialized,

        /// Weight state is frozen — no further mutations, graph replay is safe.
        /// Host data may still be present for CPU execution or future host access.
        Frozen,

        /// Host raw data released — only prepared device state remains.
        /// Cannot create new prepared handles or access raw weight data after this point.
        HostReleased,
    };

    /**
     * @brief Completion gates for weight lifecycle transitions (Phase 9)
     *
     * Each gate is a boolean flag set exactly once when the corresponding
     * lifecycle phase completes. Gates are monotonic — once set, never cleared.
     * They provide deterministic ordering for host release and freeze semantics,
     * replacing ad hoc timing comments and convention-based ordering.
     */
    struct WeightLifecycleGates
    {
        /// All source tensors loaded from GGUF file
        bool materialization_complete = false;

        /// All device-specific preparation done (GEMM packing, embedding prep, GPU upload)
        bool device_preparation_complete = false;

        /// All graph builders have resolved their weight bindings
        bool graph_materialization_complete = false;

        /// Host data release is allowed (all gates above must be true)
        bool host_release_allowed = false;

        /// Current lifecycle state (derived from gates)
        WeightLifecycleState currentState() const
        {
            if (host_release_allowed)
                return WeightLifecycleState::HostReleased;
            if (graph_materialization_complete)
                return WeightLifecycleState::Frozen;
            if (device_preparation_complete)
                return WeightLifecycleState::DevicePrepared;
            if (materialization_complete)
                return WeightLifecycleState::DerivedMaterialized;
            return WeightLifecycleState::Planned;
        }

        /// Check if host release is safe (all prerequisites met)
        bool canReleaseHostData() const
        {
            return materialization_complete &&
                   device_preparation_complete &&
                   graph_materialization_complete;
        }
    };

    const char *toString(WeightLifecycleState state);

    std::string toString(WeightRole role);
    std::string toString(WeightDerivationKind derivation);
    std::string toString(WeightHostPolicy policy);
    std::string toString(WeightResidencyCategory category);

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
