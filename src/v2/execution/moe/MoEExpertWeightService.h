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
#include "MoERebalanceController.h"  // ExpertReplicaSet
#include "../../backends/DeviceId.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace llaminar2 {

class TensorBase;
class ITensorGemm;
class ExpertWeightPayloadProvider;

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

    // GPU lifetime management
    std::vector<std::shared_ptr<ITensorGemm>>& moe_owned_kernels;
    std::shared_ptr<void>& moe_packed_gate_lifetime;
    std::shared_ptr<void>& moe_packed_up_lifetime;
    std::shared_ptr<void>& moe_packed_down_lifetime;

    // Payload provider for lazy GPU expert preparation (model-context owned).
    // When non-null, used instead of raw GGUF host data for GPU repack.
    ExpertWeightPayloadProvider* payload_provider = nullptr;
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

    // ── Weight serialization (for MPI transfer) ──────────────────────

    /// Detach and serialize packed weights for a departing expert (destructive).
    static ExpertWeightBlobs detachAndSerializeExpert(MoEWeightContext& ctx, int expert_id);

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
        const std::unordered_map<int, ExpertWeightBlobs>* received_weights);

    // ── GPU-direct transfer ──────────────────────────────────────────

    /// Transfer expert weights directly between GPU devices (GPU↔GPU memcpy).
    /// ~50x faster than serialize → MPI → deserialize → repack for intra-node
    /// transfers since both GPUs use the identical packed weight format.
    /// @param src_ctx Source MoE weight context (has packed GPU weights)
    /// @param dst_ctx Destination MoE weight context (will receive weights)
    /// @param expert_ids Experts to transfer
    /// @param layer_idx Layer index (for logging)
    /// @return true if GPU-direct transfer succeeded, false to fall back to serialize path
    static bool transferExpertsGPUDirect(
        const MoEWeightContext& src_ctx,
        MoEWeightContext& dst_ctx,
        const std::vector<int>& expert_ids,
        int layer_idx);

private:
    /// Register transferred packed weights for one expert (CPU path only).
    static bool registerTransferredExpert(MoEWeightContext& ctx, int expert_id, const ExpertWeightBlobs& blobs);

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
