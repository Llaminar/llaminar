/**
 * @file MemoryPlanner.h
 * @brief Typed admission contracts for model, graph, and routed-expert memory.
 *
 * The planner is the allocation authority before a production graph is built.
 * Its device configuration records not only model placement but also the
 * lifetime topology of graph-stable routed-expert buffers, allowing
 * captured serial graph families to share storage without under-admitting
 * concurrent graph families.
 */

#pragma once
#include "planning/MemoryPlan.h"
#include "planning/ModelMemoryProfile.h"
#include "planning/WeightMemoryEstimator.h"
#include "backends/DeviceId.h"
#include "execution/config/RuntimeConfig.h"

#include <vector>
#include <string>

namespace llaminar2
{

/** @brief Persistent execution state owned by one planned device. */
enum class DeviceExecutionMemoryRole
{
    /** Full continuation graph: KV, recurrent state, activations, and workspaces. */
    ContinuationGraph,
    /** Sparse routed-expert endpoint: no attention/KV/recurrent authority. */
    RoutedExpertParticipant,
};

/**
 * @brief Whether planned model weights require a new device allocation.
 *
 * Current free-memory readings already exclude allocations retained by the
 * process.  A second runner that adopts a plan-certified PreparedWeightStore
 * must therefore preserve the complete weight BOM while charging only its new
 * graph/runtime allocations against current free memory.
 */
enum class PreparedWeightAdmission
{
    /** The planner must reserve and allocate the complete planned weight set. */
    AllocateCompleteSet,
    /** A matching production plan certifies the complete set is already resident. */
    ReuseCertifiedCompleteSet,
};

/**
 * @brief Additional persistent weight authorities retained beside the primary graph view.
 *
 * Phase-split execution can keep a tensor-parallel prefill set and a second
 * fully replicated decode set alive at the same time. Naming that second set
 * explicitly prevents capacity planners from treating it as transient upload
 * staging or overlooking it because the primary view already contains the
 * same logical tensors in a different physical layout.
 */
enum class AdditionalPersistentWeightSet
{
    /** Full non-routed dense/shared/global decode view on each TP participant. */
    ReplicatedDenseDecode,
};

/**
 * @brief Resolve extra persistent sets implied by one dense execution policy.
 * @param policy Declarative dense/shared execution policy.
 * @param tensor_parallel_degree Number of participants in the primary view.
 * @return Complete, duplicate-free list of additional physical weight sets.
 */
[[nodiscard]] inline std::vector<AdditionalPersistentWeightSet>
resolveAdditionalPersistentWeightSets(
    DenseParallelPolicy policy,
    int tensor_parallel_degree)
{
    if (tensor_parallel_degree > 1 &&
        denseParallelPolicyReplicatesDecode(policy))
    {
        return {AdditionalPersistentWeightSet::ReplicatedDenseDecode};
    }
    return {};
}

/**
 * @brief Ownership topology for compact local-expert route tensors.
 *
 * A conventional graph retains one compact packet per participating layer.
 * Qwen ExpertOverlay instead declares a serial graph family: decode, prefill,
 * and verifier roles are explicitly ordered, so one immutable packet arena per
 * local participant is sufficient.  The planner must receive this typed fact
 * rather than inferring it from a device or model name.
 */
enum class RoutedExpertCompactBufferLifetime
{
    /** Each participating model layer owns a distinct graph-stable packet. */
    PerLayerGraphOwned,
    /** One graph-stable packet is shared by each serial local participant. */
    SerialFamilyPerParticipant,
};

/// Configuration for a single device in a memory plan.
struct DevicePlanConfig
{
    DeviceId device;
    size_t device_total_bytes = 0;
    size_t device_free_bytes = 0;

    /** CUDA SM or ROCm CU count used by capture-time launch/workspace policy. */
    int device_compute_units = 0;

    // TP configuration for this device
    int shard_index = 0;
    int total_shards = 1;

    // PP configuration: layer range
    int first_layer = 0;
    int last_layer = -1;  // -1 = all

    // KV cache configuration
    std::string kv_precision = "fp16";
    int local_kv_heads = 0;   // After TP sharding, 0 = use profile.n_kv_heads

    // Runtime parameters
    int batch_size = 1;
    int max_seq_len = 0;  // 0 = use profile.max_seq_len
    int activation_seq_len = 0;  // 0 = use max_seq_len for activation/workspace
    bool mtp_enabled = false;

    /** Flattened target-verifier rows retained by the MTP graph family. */
    int mtp_target_query_rows = 2;

    /** Exact participant-local terminal-logit ownership used by MTP. */
    MTPTerminalLogitsLayout mtp_terminal_logits_layout =
        MTPTerminalLogitsLayout::FullVocabularyPerParticipant;

    /** Exact ownership contract for compact routed-expert activation packets. */
    RoutedExpertCompactBufferLifetime routed_expert_compact_buffer_lifetime =
        RoutedExpertCompactBufferLifetime::PerLayerGraphOwned;

    /**
     * @brief Number of local participants with an independently runnable serial packet.
     *
     * Required when @ref routed_expert_compact_buffer_lifetime is
     * @ref RoutedExpertCompactBufferLifetime::SerialFamilyPerParticipant and
     * this device owns at least one routed expert.  It is deliberately a
     * count rather than a device heuristic: several logical participants may
     * share one DeviceId while still requiring non-aliasing concurrent storage.
     */
    int serial_routed_expert_participant_count = 0;

    /**
     * @brief Flattened token-row capacity of one serial compact family.
     *
     * Zero uses @ref activation_seq_len and @ref batch_size. ExpertOverlay
     * supplies its resolved captured-segment envelope here because dense graph
     * and KV capacities may be much larger. A positive value also selects the
     * production power-of-two route-family ladder through the resulting
     * `rows * top_k` maximum, so admission prices every stable address later
     * retained by the participant graph.
     */
    int serial_routed_expert_compact_rows = 0;

    /** Exact model-weight authority installed on this device. */
    DeviceWeightResidency weight_residency;

    /** Alternate physical weight views retained concurrently with the primary set. */
    std::vector<AdditionalPersistentWeightSet> additional_weight_sets;

    /** Allocation status of the exact planned weight authority. */
    PreparedWeightAdmission prepared_weight_admission =
        PreparedWeightAdmission::AllocateCompleteSet;

    /** Runtime-state role; auxiliary experts never own dense-model caches. */
    DeviceExecutionMemoryRole execution_role =
        DeviceExecutionMemoryRole::ContinuationGraph;

    // Headroom
    size_t headroom_bytes = 128ULL * 1024 * 1024;
};

/**
 * @brief Result of selecting one common resident graph-row capacity.
 *
 * GPU participants in one forward graph family must agree on the largest
 * prefill shape that can be resident at once. The full context remains owned
 * by the KV cache, while ordinary activation rows are bounded by this value.
 * Workspace families may still scale with full context when a captured kernel
 * publishes context-partition summaries. Long prompts are represented by
 * serial replays of captured bucket graphs.
 */
struct ResidentGraphMemoryPlan
{
    int resident_graph_rows = 0;
    MemoryPlan memory_plan;

    /// @brief True when the selected row capacity and full context both fit.
    bool fits() const { return resident_graph_rows > 0 && memory_plan.fits(); }
};

class MemoryPlanner
{
public:
    /// Plan memory for a set of devices with the given model profile.
    static MemoryPlan plan(
        const ModelMemoryProfile& profile,
        const std::vector<DevicePlanConfig>& device_configs
    );

    /**
     * @brief Select the largest configured graph row bucket that fits.
     *
     * Every GPU config is evaluated with the same candidate so LocalTP and
     * other symmetric graph domains cannot silently construct incompatible
     * resident shapes. CPU configs retain their requested activation length.
     * When no candidate fits, the returned plan contains the smallest
     * candidate's diagnostics and `fits()` is false.
     */
    static ResidentGraphMemoryPlan planLargestFittingResidentGraphRows(
        const ModelMemoryProfile& profile,
        const std::vector<DevicePlanConfig>& device_configs,
        const std::vector<int>& candidate_rows
    );
};

} // namespace llaminar2
