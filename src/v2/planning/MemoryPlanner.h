/**
 * @file MemoryPlanner.h
 * @brief Typed admission contracts for model, graph, and routed-expert memory.
 *
 * The planner supplies typed allocation bills to PhysicalMemoryAuthority; it
 * does not carry a parallel live ledger. Its device configuration records model
 * placement separately from rank-local collective ownership, as well as the
 * lifetime topology of graph-stable routed-expert buffers, allowing
 * captured serial graph families to share storage without under-admitting
 * concurrent graph families.
 */

#pragma once
#include "planning/MemoryPlan.h"
#include "planning/ModelMemoryProfile.h"
#include "planning/GraphSnapshotMemoryCapacity.h"
#include "planning/WeightMemoryEstimator.h"
#include "backends/DeviceId.h"
#include "config/CollectiveBackendType.h"
#include "config/TensorParallelConfig.h"
#include "execution/config/RuntimeConfig.h"
#include "execution/mtp/MTPGraphOwnerPlan.h"
#include "loaders/PreparedWeightAdmission.h"

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
    /** Full-vocabulary embedding retained beside the primary TP shard. */
    MirroredDecodeEmbedding,
    /** Full-vocabulary MTP terminal head retained beside the primary TP shard. */
    MirroredMTPTerminalHead,
    /** Complete non-routed learned MTP predictor block per TP participant. */
    ReplicatedMTPSidecarDense,
};

/**
 * @brief Resolve extra persistent sets implied by one dense execution policy.
 * @param policy Declarative dense/shared execution policy.
 * @param tensor_parallel_degree Number of participants in the primary view.
 * @param mtp Complete retained MTP graph and weight-placement policy. A
 *        mirrored terminal policy applies to the serial decode oracle even
 *        when MTP execution is disabled; a replicated predictor is retained
 *        whenever the graph-capacity envelope exists.
 * @return Complete, duplicate-free list of additional physical weight sets.
 */
[[nodiscard]] inline std::vector<AdditionalPersistentWeightSet>
resolveAdditionalPersistentWeightSets(
    DenseParallelPolicy policy,
    int tensor_parallel_degree,
    const MTPRuntimeConfig &mtp)
{
    if (tensor_parallel_degree <= 1)
        return {};
    if (denseParallelPolicyReplicatesDecode(policy))
        return {AdditionalPersistentWeightSet::ReplicatedDenseDecode};

    std::vector<AdditionalPersistentWeightSet> sets;
    const bool mirrored_mtp_head =
        mtpTerminalHeadIsMirrored(mtp.terminal_head_policy);
    if (denseParallelPolicyMirrorsDecodeEmbedding(policy))
    {
        sets.push_back(
            AdditionalPersistentWeightSet::MirroredDecodeEmbedding);
    }
    if (mirrored_mtp_head)
    {
        sets.push_back(
            AdditionalPersistentWeightSet::MirroredMTPTerminalHead);
    }
    if (retainsMTPGraphCapacity(mtp) &&
        mtp.sidecar_dense_policy ==
            MTPSidecarDensePolicy::ReplicatedPerParticipant)
    {
        sets.push_back(
            AdditionalPersistentWeightSet::ReplicatedMTPSidecarDense);
    }
    return sets;
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

/**
 * @brief Exact retained native forward-executable inventory for one device.
 *
 * Native CUDA/HIP graph executables own opaque driver allocations outside the
 * tensor and workspace arenas. The prefill ladder is capacity-dependent, while
 * decode and restored-prefix bridge executables are fixed members. Keeping the
 * two counts typed lets resident-row admission price the same family that setup
 * will later materialize without guessing from a model or backend name.
 */
struct CapturedServingGraphMemoryInventory
{
    /** Configured prefill bucket boundaries before resident-row clamping. */
    std::vector<int> prefill_bucket_rows;
    /** Complete forward executables retained independently of prefill rows. */
    std::size_t fixed_executable_count = 0u;
    /** Canonical MTP helper/controller cache geometry and graph owners. */
    MTPGraphOwnerPlan mtp_graph_owners{MTPRuntimeConfig{}};

    /** @return true when this declaration names at least one executable. */
    [[nodiscard]] bool enabled() const noexcept
    {
        return !prefill_bucket_rows.empty() ||
               fixed_executable_count != 0u ||
               mtp_graph_owners.auxiliaryExecutableSlotCount() != 0u;
    }
};

/**
 * @brief Resolve the exact ordinary serving-graph inventory retained at setup.
 * @param prefill_bucket_rows Configured captured-prefill bucket boundaries.
 * @param mtp Runtime execution and retained-capacity policy.
 * @return Inventory consumed by @ref MemoryPlanner before graph construction.
 *
 * The first request in a reusable model context may execute with MTP disabled
 * while reserving a wider graph family for later requests. Admission must
 * therefore follow retained capacity, not the current execution depth. CUDA
 * conditional branches and HIP ticket-selected transactions consume the same
 * typed cache-owner plan as runtime materialization. Semantic parent fragments
 * are intentionally absent because they do not own independent executables.
 */
[[nodiscard]] inline CapturedServingGraphMemoryInventory
resolveCapturedServingGraphMemoryInventory(
    std::vector<int> prefill_bucket_rows,
    const MTPRuntimeConfig &mtp)
{
    const bool retains_mtp = retainsMTPGraphCapacity(mtp);
    return CapturedServingGraphMemoryInventory{
        .prefill_bucket_rows = std::move(prefill_bucket_rows),
        .fixed_executable_count =
            1u +
            (retains_mtp ? 1u : 0u) +
            resolveMTPRetainedServingForwardModelGraphIdentityCount(mtp),
        .mtp_graph_owners = MTPGraphOwnerPlan(mtp),
    };
}

/// Configuration for a single device in a memory plan.
struct DevicePlanConfig
{
    /** MPI rank owning this allocator; -1 is valid for unbound pure estimates. */
    int world_rank = -1;
    DeviceId device;
    size_t device_total_bytes = 0;
    size_t device_free_bytes = 0;

    /**
     * Host allocator that owns CPU-side state created by this execution graph.
     * A GPU prefix cache, for example, retains a bounded RAM tier in addition
     * to its VRAM staging/device tier. CPU graphs leave this empty because
     * their primary @ref device already names the same physical authority.
     */
    std::optional<PhysicalMemoryResource> associated_host_memory;

    /**
     * @brief Exact transient weight-load allocations for this GPU.
     *
     * The device and pinned-host rings are separate physical owners even
     * though they share one upload geometry. CPU plans and certified retained
     * weights leave both values at zero. A non-zero host value requires
     * @ref associated_host_memory so it can be charged to the rank-local CPU
     * allocator rather than hidden in the GPU subtotal.
     */
    struct WeightLoadStaging
    {
        std::size_t device_bytes = 0u;
        std::size_t host_bytes = 0u;

        /** @return Whether either physical side owns a transient allocation. */
        [[nodiscard]] bool enabled() const noexcept
        {
            return device_bytes != 0u || host_bytes != 0u;
        }
    } weight_load_staging;

    /**
     * CUDA SMs, ROCm CUs, or configured physical-core CPU workers used by the
     * exact runtime launch/workspace policy.
     */
    int device_compute_units = 0;

    // TP configuration for this device
    int shard_index = 0;
    int total_shards = 1;

    /**
     * Optional resolved rank-local collective backend for this participant.
     *
     * Absence means this rank installs no LocalTP context. Distributed TP may
     * still shard weights across ranks, so total_shards cannot establish this
     * allocation's presence. A present value must name the concrete backend
     * installed by LocalTP; AUTO is invalid, not an alias for absence.
     */
    std::optional<CollectiveBackendType> local_tp_backend;

    /**
     * Exact tensor-parallel slice installed by the production assignment authority.
     *
     * Rank-local TP may use uneven query/FFN/vocabulary ranges and may
     * replicate GQA KV heads. `shard_index / total_shards` therefore cannot
     * reconstruct the participant geometry. Cross-rank uniform TP may leave
     * this empty because its validator proves exact divisibility.
     */
    std::optional<DeviceShardingAssignment> tensor_parallel_assignment;

    /**
     * @brief Bind the exact production tensor-parallel assignment.
     * @param assignment Assignment produced by @ref TensorParallelConfig.
     * @throws std::invalid_argument when it names another device/rank or an
     *         invalid local slice.
     */
    void bindTensorParallelAssignment(
        const DeviceShardingAssignment &assignment)
    {
        if (total_shards <= 1 || assignment.local_rank != shard_index ||
            assignment.device != device || !assignment.isValid() ||
            assignment.kv_head_count <= 0 ||
            assignment.d_ff_count <= 0 ||
            assignment.vocab_count <= 0)
        {
            throw std::invalid_argument(
                "Device memory plan tensor-parallel assignment does not match its physical shard identity");
        }
        tensor_parallel_assignment = assignment;
        local_kv_heads = assignment.kv_head_count;
    }

    // PP configuration: layer range
    int first_layer = 0;
    int last_layer = -1;  // -1 = all
    /** Whether this participant owns the graph-captured token embedding. */
    bool owns_embedding = true;

    // KV cache configuration
    std::string kv_precision = "fp16";
    int local_kv_heads = 0;   // After TP sharding, 0 = use profile.n_kv_heads

    /**
     * Prefix-state archive policy retained by this continuation device.
     *
     * GPU archive staging and the bounded device-hot tier are real concurrent
     * owners beside the live KV cache. Keeping the production policy in the
     * device plan lets auto-capacity and final preflight price the same bytes.
     * Routed-expert-only participants ignore this field because they own no KV
     * or recurrent inference state.
     */
    PrefixCacheRuntimeConfig prefix_cache = []
    {
        PrefixCacheRuntimeConfig disabled;
        disabled.enabled = false;
        disabled.storage_mode = PrefixCacheStorageMode::Disabled;
        return disabled;
    }();

    // Runtime parameters
    int batch_size = 1;
    int max_seq_len = 0;  // 0 = use profile.max_seq_len
    int activation_seq_len = 0;  // 0 = use max_seq_len for activation/workspace
    bool mtp_enabled = false;

    /**
     * Exact physical KV-head ownership of the retained shifted MTP cache.
     * This is resolved from the same sidecar policy used by graph construction;
     * memory admission must not infer it from the main cache's TP slice.
     */
    MTPShiftedKVHeadLayout mtp_shifted_kv_head_layout =
        MTPShiftedKVHeadLayout::PrimaryTensorParallelShard;

    /** Flattened target-verifier rows retained by the MTP graph family. */
    int mtp_target_query_rows = 2;

    /** Exact participant-local terminal-logit ownership used by MTP. */
    MTPTerminalLogitsLayout mtp_terminal_logits_layout =
        MTPTerminalLogitsLayout::FullVocabularyPerParticipant;

    /** Exact ownership contract for compact routed-expert activation packets. */
    RoutedExpertCompactBufferLifetime routed_expert_compact_buffer_lifetime =
        RoutedExpertCompactBufferLifetime::PerLayerGraphOwned;

    /** Native graph family whose opaque driver storage must be admitted. */
    CapturedServingGraphMemoryInventory captured_serving_graphs;

    /**
     * Diagnostic checkpoint storage retained beside captured executables.
     * Zero is valid only when graph snapshots are disabled. The caller that
     * selects a snapshot topology owns this complete per-accelerator bound;
     * MemoryPlanner merely converts it into the canonical physical BOM.
     */
    GraphSnapshotMemoryCapacity graph_snapshot_memory;

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

    /**
     * Primary workspace bytes already retained by the exact prepared-model
     * context. The live free-memory observation excludes these bytes, while a
     * newly planned serial family can republish them without allocation.
     */
    std::size_t retained_workspace_bytes = 0u;

    /** Runtime-state role; auxiliary experts never own dense-model caches. */
    DeviceExecutionMemoryRole execution_role =
        DeviceExecutionMemoryRole::ContinuationGraph;

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
     * resident shapes. A retained MTP verifier family can raise that common
     * capacity above the selected prefill bucket; the returned row count and
     * byte plan always describe the complete prefill-plus-MTP graph family.
     * Ordinary CPU configs retain their requested activation length, while CPU
     * routed-expert participants share the captured ticket geometry. When no
     * candidate fits, the returned plan contains the smallest candidate's
     * diagnostics and `fits()` is false.
     */
    static ResidentGraphMemoryPlan planLargestFittingResidentGraphRows(
        const ModelMemoryProfile& profile,
        const std::vector<DevicePlanConfig>& device_configs,
        const std::vector<int>& candidate_rows
    );
};

} // namespace llaminar2
