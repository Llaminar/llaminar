/**
 * @file MoEExpertWeightService.h
 * @brief Weight lifecycle service for MoE expert GEMM engines.
 *
 * Extracted from MoEExpertComputeStage to separate weight preparation, serialization,
 * and rebalancing concerns from the compute stage. Called at graph-build time
 * and during dynamic rebalancing — NOT during inference execution.
 *
 * All methods are static (stateless service). State lives in MoEWeightContext
 * which references MoEExpertComputeStage::Params fields.
 */

#pragma once

#include "ExpertWeightTransfer.h"    // ExpertWeightBlobs
#include "GPUExpertTransfer.h"
#include "MoERebalanceController.h"  // ExpertReplicaSet
#include "../../backends/BackendManager.h"
#include "../../backends/DeviceId.h"
#include "../../loaders/ExpertSlabTypes.h"

#include <memory>
#include <optional>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace llaminar2 {

class TensorBase;
class ITensorGemm;
class ExpertWeightPayloadProvider;
class PreparedWeightStore;
class ExpertGemmRegistry;
class GpuExpertSlotPool;
class GpuExpertTransferStagingPool;

/**
 * @brief Typed NUMA ownership policy for persistent CPU expert weights.
 *
 * A weight context must state whether CPU placement is irrelevant, spans an
 * aggregate CPU domain, or belongs to one exact NUMA node. The deleted default
 * constructor makes adding a new context construction site without making that
 * decision a compile-time error.
 */
class CPUExpertNUMAPlacement
{
public:
    enum class Scope
    {
        NotApplicable,
        AggregateDomain,
        BoundNode,
    };

    CPUExpertNUMAPlacement() = delete;

    /** @brief Construct policy for a non-CPU weight context. */
    static CPUExpertNUMAPlacement notApplicable()
    {
        return CPUExpertNUMAPlacement(Scope::NotApplicable, -1);
    }

    /** @brief Construct policy for one process spanning the aggregate CPU domain. */
    static CPUExpertNUMAPlacement aggregateDomain()
    {
        return CPUExpertNUMAPlacement(Scope::AggregateDomain, -1);
    }

    /**
     * @brief Construct strict ownership by one exact NUMA node.
     * @throws std::invalid_argument when @p node is negative.
     */
    static CPUExpertNUMAPlacement boundNode(int node)
    {
        if (node < 0)
            throw std::invalid_argument(
                "Bound CPU expert NUMA placement requires a non-negative node");
        return CPUExpertNUMAPlacement(Scope::BoundNode, node);
    }

    /**
     * @brief Resolve policy from the rank-local backend declaration.
     *
     * GPU contexts are explicitly not applicable. CPU contexts use the NUMA
     * node captured when the rank's `CPUBackend` was initialized; an aggregate
     * backend remains explicit rather than guessing from the current thread.
     */
    static CPUExpertNUMAPlacement forDevice(DeviceId device)
    {
        if (!device.is_cpu())
            return notApplicable();
        const int node = cpuBackendNUMANode();
        return node >= 0 ? boundNode(node) : aggregateDomain();
    }

    /** @brief Return the declared placement scope. */
    Scope scope() const noexcept { return scope_; }

    /** @brief Return whether every persistent packed page must reside on one node. */
    bool requiresNodeBinding() const noexcept
    {
        return scope_ == Scope::BoundNode;
    }

    /** @brief Return the exact node, or `-1` when no single node is declared. */
    int node() const noexcept { return node_; }

private:
    CPUExpertNUMAPlacement(Scope scope, int node) noexcept
        : scope_(scope), node_(node)
    {
    }

    Scope scope_;
    int node_;
};

/// Lightweight reference struct pointing to the MoEExpertComputeStage::Params fields
/// that the weight service operates on. Avoids coupling the service to the
/// full Params struct.
struct MoEWeightContext {
    DeviceId device_id;
    int num_experts = 0;
    int expert_intermediate = 0;
    int d_model = 0;
    int local_expert_start = 0;
    int local_expert_count = -1;
    int layer_idx = -1;
    std::vector<bool>& expert_mask;

    // 3D packed parent tensors (may be null after releaseRawWeights)
    TensorBase* gate_exps = nullptr;
    TensorBase* up_exps = nullptr;
    TensorBase* down_exps = nullptr;

    // Per-expert 2D views [num_experts] each
    std::vector<std::shared_ptr<TensorBase>>& expert_gate_views;
    std::vector<std::shared_ptr<TensorBase>>& expert_up_views;
    std::vector<std::shared_ptr<TensorBase>>& expert_down_views;

    // Pre-resolved GEMM engines per expert [num_experts]
    std::vector<ITensorGemm*>& prepared_gate_gemm;
    std::vector<ITensorGemm*>& prepared_up_gemm;
    std::vector<ITensorGemm*>& prepared_down_gemm;

    // Engine lifetime management. For store-backed CPU initial prep, ownership
    // is handed to PreparedWeightStore and this vector is cleared after registration.
    std::vector<std::shared_ptr<ITensorGemm>>& moe_owned_kernels;
    std::shared_ptr<void>& moe_packed_gate_lifetime;
    std::shared_ptr<void>& moe_packed_up_lifetime;
    std::shared_ptr<void>& moe_packed_down_lifetime;

    // Payload provider for runtime GPU expert arrivals (model-context owned).
    // When non-null, used instead of raw GGUF host data for GPU repack.
    ExpertWeightPayloadProvider* payload_provider = nullptr;

    // Prepared expert lifetime store. When non-null, store-backed initial CPU
    // prep hands engine ownership to this store so cached graphs only keep raw refs.
    PreparedWeightStore* prepared_store = nullptr;

    // ExpertGemmRegistry for dynamic rebalancing registry updates.
    // When non-null, engine arrivals/departures are mirrored to the registry.
    ExpertGemmRegistry* expert_registry = nullptr;

    // Phase C: Cached slab refs (set by prepareGemmEngines, reused by rebalance).
    // Optional because they're only populated when prepared_store is non-null.
    std::optional<ExpertSlabRef> gate_slab_ref;
    std::optional<ExpertSlabRef> up_slab_ref;
    std::optional<ExpertSlabRef> down_slab_ref;

    // Initial prep may share mmap-backed parent tensors across accelerator and
    // CPU fallback tiers. Disable eager page advice until every consumer is done.
    bool advise_raw_pages_after_prepare = true;

    // Optional per-stage dynamic GPU arrival pool. When present, same-backend
    // GPU-direct arrivals can reuse physical expert slots instead of creating a
    // fresh VRAM allocation for every rebalanced expert.
    std::shared_ptr<GpuExpertSlotPool>* gpu_direct_slot_pool = nullptr;

    // Required typed policy: omission at a new construction site is a compile
    // error instead of silently disabling strict arrival placement.
    CPUExpertNUMAPlacement cpu_numa_placement;

};

/// One projection of a GPU-direct expert arrival that has been copied into
/// staging/active slots but has not yet been published to live stage tables.
struct GpuDirectStagedExpertProjection
{
    int expert_id = -1;
    WeightRole role = WeightRole::Other;
    ITensorGemm* engine = nullptr;
    std::shared_ptr<ITensorGemm> engine_lifetime;
    std::optional<GpuExpertStagedActivation> activation;

    bool valid() const
    {
        return expert_id >= 0 &&
               engine != nullptr &&
               (role == WeightRole::MoEExpertGate ||
                role == WeightRole::MoEExpertUp ||
                role == WeightRole::MoEExpertDown);
    }
};

/// Background-safe carrier for staged GPU-direct expert arrivals.
///
/// Building this object must not mutate MoEExpertComputeStage prepared-engine
/// vectors. Activation/publish code consumes it later on the runner thread.
struct GpuDirectStagedExpertArrivals
{
    DeviceId device_id;
    std::optional<DeviceId> source_device;
    int layer_idx = -1;
    std::vector<GpuDirectStagedExpertProjection> projections;
    GpuDirectTransferCompletion completion;

    bool empty() const { return projections.empty(); }
    size_t projectionCount() const { return projections.size(); }
    size_t activationCount() const;
    std::vector<int> expertIds() const;
};

/// One projection copied into a surplus transfer slot.
///
/// This is deliberately not publishable: it owns scratch/staging metadata only.
/// Runner-thread activation must allocate an active slot, enqueue the same-device
/// slot copy, wrap active-slot pointers in a backend GEMM engine, and only then
/// publish a GpuDirectStagedExpertProjection.
struct GpuDirectTransferSlotProjection
{
    int expert_id = -1;
    WeightRole role = WeightRole::Other;
    GpuExpertPackedDescriptor staged;
    int N = 0;
    int K = 0;
    uint32_t blocks_per_row = 0;
    uint8_t payload_bytes_per_block = 0;
    bool is_asymmetric = false;
    bool has_emins = false;
    uint8_t codebook_id = 0;
    NativeVnniSourceIdentity source_identity;
    std::shared_ptr<void> transfer_slot_lifetime;

    bool valid() const;
    size_t bytes() const { return staged.totalBytes(); }
};

/// Background-safe carrier for expert projections staged in transfer slots.
///
/// Building this object may enqueue GPU copies and hold transfer-slot leases, but
/// must not mutate live MoE stage tables, active slot ownership, or the prepared
/// store. It is consumed later on the runner thread.
struct GpuDirectTransferSlotArrivals
{
    DeviceId device_id;
    std::optional<DeviceId> source_device;
    int layer_idx = -1;
    std::vector<GpuDirectTransferSlotProjection> projections;
    GpuDirectTransferCompletion completion;

    bool empty() const { return projections.empty(); }
    size_t projectionCount() const { return projections.size(); }
    size_t totalBytes() const;
    std::vector<int> expertIds() const;
    std::vector<std::shared_ptr<void>> transferSlotLifetimes() const;
};

/// Weight lifecycle service for MoE expert GEMM engines.
///
/// Extracted from MoEExpertComputeStage to separate weight preparation, serialization,
/// and rebalancing concerns from the compute stage. Called at graph-build time
/// and during dynamic rebalancing — NOT during inference execution.
///
/// All methods are static (stateless service). State lives in MoEWeightContext
/// which references MoEExpertComputeStage::Params fields.
class MoEExpertWeightService {
public:
    // ── Graph-build time ─────────────────────────────────────────────

    /// Extract 2D expert views from 3D packed tensors.
    /// Call once at graph-build time. Views stored in ctx.expert_*_views.
    static bool extractExpertViews(MoEWeightContext& ctx);

    /// Prepare GEMM engines for all expert views.
    /// Must be called after extractExpertViews(). Dispatches to CPU/CUDA/ROCm.
    static bool prepareGemmEngines(MoEWeightContext& ctx);

    /// Release 3D parent weight tensors to free raw (un-packed) weight memory.
    /// Returns bytes freed.
    static size_t releaseRawWeights(MoEWeightContext& ctx);

    // ── Prepared-weight movement and serialization ───────────────────

    /**
     * @brief Detach the three final packed CPU projections for ownership movement.
     *
     * This is O(1): the packed allocations move out of their source engines and
     * are subsequently consumed by direct MPI section transfer.
     */
    static ExpertPackedWeights detachPreparedExpert(
        MoEWeightContext& ctx,
        int expert_id);

    /**
     * @brief Clone the three final packed CPU projections for replication.
     *
     * The source engines remain resident. The explicit clone is the ownership
     * cost required by a replica; it is not followed by another serialization
     * copy on the homogeneous CPU transfer path.
     */
    static ExpertPackedWeights clonePreparedExpert(
        const MoEWeightContext& ctx,
        int expert_id);

    /// Serialize packed weights for an expert without detaching (non-destructive).
    static ExpertWeightBlobs serializeExpert(const MoEWeightContext& ctx, int expert_id);

    // ── Phased rebalance API ─────────────────────────────────────────

    /// Phase 1: Release departed expert engines, return tensor views to evict.
    static std::vector<const TensorBase*> releaseDepartedExperts(
        MoEWeightContext& ctx, const std::vector<bool>& new_mask);

    /// Phase 2: Register transferred weights and prepare GEMM engines.
    static bool registerAndPrepareNewExperts(
        MoEWeightContext& ctx,
        const std::vector<bool>& new_mask,
        const std::unordered_map<int, ExpertWeightBlobs>* received_weights,
        const std::unordered_map<int, PreparedExpertEngines>*
            received_prepared_experts = nullptr);

    // ── GPU-direct transfer ──────────────────────────────────────────

    /// Transfer expert weights directly between GPU devices (GPU↔GPU memcpy).
    /// ~50x faster than serialize → MPI → deserialize → repack for intra-node
    /// transfers since both GPUs use the identical packed weight format.
    /// @param src_ctx Source MoE weight context (has packed GPU weights)
    /// @param dst_ctx Destination MoE weight context (will receive weights)
    /// @param expert_ids Experts to transfer
    /// @param layer_idx Layer index (for logging)
    /// @param source_producer_stream Explicit source GPU stream. The destination
    /// transfer stream waits on an event recorded here before copying.
    /// @param satisfied_expert_ids Optional output populated with requested experts
    /// that are resident on the destination after the attempt. Includes experts
    /// that were already resident before the peer copy.
    /// @return true if GPU-direct transfer succeeded, false to fall back to serialize path
    static bool transferExpertsGPUDirect(
        const MoEWeightContext& src_ctx,
        MoEWeightContext& dst_ctx,
        const std::vector<int>& expert_ids,
        int layer_idx,
        void* source_producer_stream,
        std::vector<int>* satisfied_expert_ids = nullptr,
        GpuDirectTransferCompletion* completion = nullptr);

    /// Stage expert weights directly into surplus GPU transfer slots.
    ///
    /// This is the async prepare phase for same-backend local GPU rebalance:
    /// source packed descriptors are copied into destination transfer slots and
    /// a readiness event is recorded, but live prepared-engine tables, active
    /// slots, registries, stores, and masks are not mutated.
    static bool stageExpertsGPUDirectToTransferSlots(
        const MoEWeightContext& src_ctx,
        MoEWeightContext& dst_ctx,
        const std::vector<int>& expert_ids,
        int layer_idx,
        void* source_producer_stream,
        GpuDirectTransferSlotArrivals* staged_arrivals,
        std::vector<int>* satisfied_expert_ids = nullptr,
        size_t active_arrival_capacity = 0,
        size_t staging_pool_capacity = 0,
        std::vector<std::shared_ptr<GpuExpertTransferStagingPool>>* transfer_staging_pools = nullptr);

    static std::vector<GpuExpertStagedActivation> activationBatchForStagedArrivals(
        const GpuDirectStagedExpertArrivals& arrivals);

    /// Activate transfer-slot arrivals into active expert slots on an explicit
    /// destination stream. The returned staged arrivals are ready to install via
    /// installActivatedGpuDirectArrivals().
    static bool activateGpuDirectTransferSlotArrivals(
        MoEWeightContext& ctx,
        const GpuDirectTransferSlotArrivals& arrivals,
        void* activation_stream,
        GpuDirectStagedExpertArrivals* activated_arrivals,
        GpuDirectTransferCompletion* completion = nullptr);

    static bool installActivatedGpuDirectArrivals(
        MoEWeightContext& ctx,
        const GpuDirectStagedExpertArrivals& arrivals);

private:
    /// GPU pipeline path: raw H2D + GPU repack via LoadOrchestrator.
    /// Unified for CUDA and ROCm — no CPU-side VNNI interleaving.
    static bool prepareGemmEnginesGPU(MoEWeightContext& ctx);

    /// GPU rebalance path: repack new experts on GPU via LoadOrchestrator.
    /// Called by registerAndPrepareNewExperts() for GPU devices instead of
    /// the CPU KernelFactory fallback.
    static bool registerAndPrepareNewExpertsGPU(
        MoEWeightContext& ctx,
        const std::vector<int>& new_experts,
        const std::unordered_map<int, ExpertWeightBlobs>* received_weights);


};

} // namespace llaminar2
