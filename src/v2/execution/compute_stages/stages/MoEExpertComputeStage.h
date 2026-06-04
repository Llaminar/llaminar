/**
 * @file MoEExpertComputeStage.h
 * @brief Unified MoE FFN stage: route → expert SwiGLU → combine
 *
 * Implements the full MoE feed-forward block as a single stage:
 * 1. Router: hidden × gate_weights → softmax → top-k selection
 * 2. Expert FFN: per-expert SwiGLU (gate, up, down) with gather/scatter
 * 3. Combine: weighted sum of expert outputs
 *
 * This is implemented as a single stage because routing creates dynamic
 * control flow that cannot be expressed as a static compute graph.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../memory/BufferId.h"
#include "../../../kernels/IMoEKernel.h"
#include "../../../loaders/WeightPlan.h"
#include "../../../loaders/ExpertSlabTypes.h"
#include "../../moe/ExpertWeightTransfer.h"
#include "../../moe/MoERebalanceController.h"
#include "../../moe/MoEExpertWeightService.h"
#include "../../moe/MoERuntimeTable.h"

#include <memory>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class ITensorGemm;
    class FP32Tensor;
    class DecodeExpertHistogram;
    class ExpertWeightPayloadProvider;
    class PreparedWeightStore;
    class ExpertGemmRegistry;

    /**
     * @brief Unified MoE FFN stage (router + expert execution + combine)
     *
     * Supports CPU, CUDA, and ROCm backends:
     * - CPU: Inline dequantization + scalar dot products (original path)
     * - GPU: Per-expert 2D tensor views → KernelFactory GEMM dispatch
     *
     * Expert views are pre-extracted at graph build time to avoid
     * runtime 3D tensor slicing overhead.
     */
    class MoEExpertComputeStage : public IComputeStage, public IWorkspaceConsumer
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            // Input
            TensorBase *input = nullptr; ///< Normalized hidden [seq_len, d_model]
            int seq_len = 0;
            int d_model = 0;

            // Router config (routing done externally by MoERoutingStage)
            int num_experts = 0;
            int top_k = 0;

            // Expert weights (3D packed tensors) — used by CPU path
            TensorBase *gate_exps = nullptr; ///< [num_experts, intermediate, d_model]
            TensorBase *up_exps = nullptr;   ///< [num_experts, intermediate, d_model]
            TensorBase *down_exps = nullptr; ///< [num_experts, d_model, intermediate]
            int expert_intermediate = 0;

            // Expert Parallelism (EP): partition experts across TP ranks.
            // When active, this rank only computes experts in
            // [local_expert_start, local_expert_start + local_expert_count).
            // Set by graph builder when TP degree > 1.
            // -1 means all experts (no EP, single-device mode).
            int local_expert_start = 0;
            int local_expert_count = -1;

            // Layer index (used by DGO for layer identification)
            int layer_idx = -1;

            /// Per-expert active mask for dynamic rebalancing.
            /// When non-empty (size == num_experts), expert_mask[e] == true means
            /// this rank should compute expert e. Overrides local_expert_start/count.
            /// When empty, falls back to contiguous range behavior.
            /// When replicas are active, includes both owned and replicated experts.
            std::vector<bool> expert_mask;

            /// Expert replication for per-token dynamic dispatch.
            /// When set (num_replicated > 0), replicated experts are assigned
            /// to sockets per-token to balance load. Both sockets have GEMM
            /// engines for replicated experts; only one computes each per token.
            ExpertReplicaSet replica_set;

            /// This rank's socket ID (for per-token replica dispatch).
            int my_socket_id = 0;

            // Per-expert 2D tensor views — used by GPU path
            // Each vector has num_experts entries; each entry is a 2D view
            // into the corresponding 3D packed tensor.
            // Set by graph builder via extractExpertViews().
            std::vector<std::shared_ptr<TensorBase>> expert_gate_views; ///< [intermediate, d_model] per expert
            std::vector<std::shared_ptr<TensorBase>> expert_up_views;   ///< [intermediate, d_model] per expert
            std::vector<std::shared_ptr<TensorBase>> expert_down_views; ///< [d_model, intermediate] per expert

            // Pre-resolved GEMM engines per expert — set by prepareExpertGemmEngines()
            // at graph build time so that execute() never triggers weight repacking.
            std::vector<ITensorGemm *> prepared_gate_gemm; ///< [num_experts] GEMM engines
            std::vector<ITensorGemm *> prepared_up_gemm;   ///< [num_experts] GEMM engines
            std::vector<ITensorGemm *> prepared_down_gemm; ///< [num_experts] GEMM engines

            // MoE batch-packed GPU lifetime management:
            // owned_kernels keeps MoE batch-constructed kernels alive,
            // packed_*_lifetime keeps the shared GPU allocation alive.
            std::vector<std::shared_ptr<ITensorGemm>> moe_owned_kernels;
            std::shared_ptr<void> moe_packed_gate_lifetime;
            std::shared_ptr<void> moe_packed_up_lifetime;
            std::shared_ptr<void> moe_packed_down_lifetime;

            // ExpertGemmRegistry for dynamic rebalancing registry updates.
            // Set by graph builder when model_ctx is available.
            ExpertGemmRegistry *expert_registry = nullptr;

            // Scratch buffers for GPU expert execution
            TensorBase *gate_scratch = nullptr; ///< [seq_len, intermediate] FP32 scratch
            TensorBase *up_scratch = nullptr;   ///< [seq_len, intermediate] FP32 scratch

            // Routing results (from MoERoutingStage)
            TensorBase *routing_indices = nullptr; ///< FP32 [seq_len * top_k] expert IDs as float
            TensorBase *routing_weights = nullptr; ///< FP32 [seq_len * top_k] normalized weights
            BufferId routing_indices_buffer_id = BufferId::MOE_EXPERT_INDICES;
            BufferId routing_weights_buffer_id = BufferId::MOE_EXPERT_WEIGHTS;

            // Output
            TensorBase *output = nullptr; ///< Combined output [seq_len, d_model]

            // Buffer IDs for coherence
            BufferId input_buffer_id = BufferId::NORMALIZED;
            BufferId output_buffer_id = BufferId::MOE_COMBINED_OUTPUT;
            bool output_registered_in_arena = true;

            // =================================================================
            // Phase 7: PreparedWeightStore for decode-time fallback resolution
            // =================================================================
            PreparedWeightStore *prepared_store = nullptr;

            // Stable graph-facing MoE runtime placement state. Owned by the graph/model
            // layer; stages only cache the per-layer device pointer.
            IMoERuntimeTable *moe_runtime_table = nullptr;

            // Phase C: Cached slab refs for store-based resolution and rebalance
            std::optional<ExpertSlabRef> gate_slab_ref;
            std::optional<ExpertSlabRef> up_slab_ref;
            std::optional<ExpertSlabRef> down_slab_ref;
        };

        explicit MoEExpertComputeStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        bool validatePreparedWeights(std::string *error) const override;
        ComputeStageType type() const override { return ComputeStageType::MOE_EXPERT_FFN; }
        std::string name() const override { return "moe_ffn"; }
        size_t estimatedFlops() const override;

        /// Layer index this stage belongs to (-1 if unset).
        int layerIndex() const { return params_.layer_idx; }

        /// In expert-parallel mode, a rank's MoE FFN output can be all zeros
        /// when no selected experts fall in its local range. The downstream
        /// AllReduce combines partial results across ranks.
        bool allowsZeroOutput() const override
        {
            return params_.local_expert_count >= 0 || !params_.expert_mask.empty();
        }

        /// Update expert mask for dynamic rebalancing (runtime, no rebuild needed).
        /// mask.size() must == num_experts. Returns false on size mismatch.
        bool updateExpertMask(const std::vector<bool> &mask);

        /// Set replica info for per-token dynamic dispatch.
        void setReplicaSet(const ExpertReplicaSet &replicas, int socket_id)
        {
            params_.replica_set = replicas;
            params_.my_socket_id = socket_id;
            // Pre-build prefill mask: single-lookup replaces multi-branch check
            if (replicas.num_replicated > 0 && !params_.expert_mask.empty())
                params_.replica_set.buildPrefillMask(socket_id, params_.expert_mask);
        }

        /// Detach and serialize packed weights for a departing expert.
        /// Returns serialized gate/up/down blobs. After this call, the expert's
        /// GEMM engines have empty weights (will be cleaned up in Phase 1 of
        /// updateExpertMaskAndPrepareEngines).
        ExpertWeightBlobs detachAndSerializeExpert(int expert_id);

        /// Serialize packed weights for an expert without detaching.
        /// The owner keeps its GEMM engines intact. Used for replica transfers
        /// where both sockets need the weights.
        ExpertWeightBlobs serializeExpert(int expert_id) const;

        // ── Phased rebalance API (used by DeviceGraphOrchestrator) ───────
        //
        // These replace the monolithic updateExpertMaskAndPrepareEngines()
        // when the caller needs to batch cache eviction across many stages.

        /// Phase 1: Release departed expert engines, return tensor views to evict.
        /// Releases packed weights and nulls engine pointers for experts that are
        /// NOT in new_mask but currently prepared.  Does NOT touch KernelFactory
        /// caches — the caller must batch-evict the returned pointers.
        std::vector<const TensorBase *> releaseDepartedExperts(
            const std::vector<bool> &new_mask);

        /// Phase 2: Register transferred weights and prepare GEMM engines for
        /// newly-acquired experts.  Call AFTER batch cache eviction of departed
        /// tensor views.
        bool registerAndPrepareNewExperts(
            const std::vector<bool> &new_mask,
            const std::unordered_map<int, ExpertWeightBlobs> *received_weights);

        /// Phase 3: Apply the new expert mask and invalidate cached engine vectors.
        void applyExpertMask(const std::vector<bool> &new_mask);

        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override;
        bool supportsWarmupDependentGraphCapture() const override;
        bool supportsPaddedPrefillGraphCapturePreflight() const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

        /// Extract 2D expert views from 3D packed tensors.
        /// Call once at graph-build time. Views are stored in params.
        static bool extractExpertViews(Params &params);

        /// Prepare GEMM engines for all expert views at graph-build time.
        /// Must be called after extractExpertViews(). Triggers VNNI repacking
        /// during model loading rather than on first inference call.
        static bool prepareExpertGemmEngines(Params &params);

        /// Release 3D parent weight tensors to free raw (un-packed) weight memory.
        /// After this call, expert views remain as KernelFactory cache keys but
        /// fallback VNNI repacking from raw data is no longer possible.
        /// Only call after all engines are prepared AND prepacked MPI transfer
        /// is available as the sole weight transfer mechanism.
        /// @return Bytes freed (approximate, from 3D tensor data)
        size_t releaseRawExpertWeights();

        /// Build a MoEWeightContext referencing this stage's params.
        /// Used by the weight service for rebalancing operations.
        MoEWeightContext buildWeightContext();

        /// Set the payload provider for runtime GPU expert arrivals.
        /// The provider is model-context owned and outlives the stage.
        void setPayloadProvider(ExpertWeightPayloadProvider *provider)
        {
            payload_provider_ = provider;
        }

        // =====================================================================
        // IWorkspaceConsumer Implementation
        // =====================================================================
        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;
        void unbindWorkspace() override;
        bool hasWorkspace() const override;
        DeviceWorkspaceManager *getWorkspace() const override;

        // Test accessor
        void setMoEKernelForTesting(IMoEKernel *kernel) { moe_kernel_ = kernel; }

    private:
        Params params_;
        bool raw_weights_released_ = false;                       ///< Set by releaseRawExpertWeights()
        DeviceWorkspaceManager *bound_workspace_ = nullptr;       ///< Workspace for expert GEMM engines
        ExpertWeightPayloadProvider *payload_provider_ = nullptr; ///< Model-context owned

        /// Cached GEMM engines per expert (resolved on first execute)
        mutable std::vector<ITensorGemm *> cached_gate_gemm_;
        mutable std::vector<ITensorGemm *> cached_up_gemm_;
        mutable std::vector<ITensorGemm *> cached_down_gemm_;

        /// Reusable scratch tensors (allocated on first use, grown if needed)
        mutable std::shared_ptr<FP32Tensor> scratch_batch_;
        mutable std::shared_ptr<FP32Tensor> scratch_gate_;
        mutable std::shared_ptr<FP32Tensor> scratch_up_;
        mutable std::shared_ptr<FP32Tensor> scratch_out_;
        mutable int scratch_capacity_ = 0;

        /// Batched gate+up scratch buffers for M=1 decode (one per top-k expert).
        /// Enables fusing all experts' gate+up into a single OMP region.
        mutable std::vector<std::shared_ptr<FP32Tensor>> scratch_gate_batch_;
        mutable std::vector<std::shared_ptr<FP32Tensor>> scratch_up_batch_;

        /// Per-expert down projection output buffers for fused Phase 2.
        mutable std::vector<std::shared_ptr<FP32Tensor>> scratch_down_batch_;

        /// Per-expert SwiGLU scratch buffers for fused Phase 2.
        mutable std::vector<std::vector<float>> swiglu_scratch_batch_;

        /// Reusable projection descriptor vector (avoids per-call heap alloc)
        mutable std::vector<ITensorGemm::TensorProjectionDesc> batch_projections_;

        /// Reusable expert-id list for full-local grouped decode descriptor preparation.
        mutable std::vector<int> all_expert_ids_;

        /// Cached MoE kernel (gather/scatter, SwiGLU fallback)
        mutable IMoEKernel *moe_kernel_ = nullptr;

        /// Fast path for decode (seq_len=1): avoids token grouping, gather/scatter,
        /// and per-expert heap allocations. Uses routing results directly.
        bool executeSingleToken(IDeviceContext *ctx);

        void ensureGemmEnginesCached();
        bool ensureGemmEnginesForExperts(const std::vector<int> &expert_ids);
        bool ensureGroupedGateUpDescriptorTable(IMoEKernel *kernel, int d_model, int intermediate);
        bool ensureGroupedDownDescriptorTable(IMoEKernel *kernel, int d_model, int intermediate);
        bool initializeMoERuntimeTableForGroupedDecode();
        bool initializeMoERuntimeTableForGroupedPrefill();
        bool initializeFixedTopologyGroupedPrefill();
        bool runtimeTableHasActiveGroupedDecodeBank() const;
        bool canUseRuntimePrefillGrouping() const;
        bool canUseFixedTopologyGroupedPrefill() const;
        bool executeFixedTopologyGroupedPrefill(IMoEKernel *kernel, int max_tokens) const;
        bool isDeviceRoutedDecodeGraphCapturable() const;
        bool supportsFixedTopologyPrefillGraphCapturePreflight() const;
        bool isFixedTopologyPrefillGraphCapturable() const;
        bool hasFullLocalExpertOwnership() const;
        bool expertMaskAllEnabled() const;
        bool hasAllPreparedExpertGemmEngines() const;
        bool hasGroupedDecodeDescriptorExportSupport() const;
        const DeviceMoEPlacementBank *activeRuntimePlacementBank() const;
        bool runtimeLocalComputeEnabled(const DeviceMoEPlacementBank *bank, int expert_id) const;
        void ensureScratchBuffers(int max_batch) const;
        IMoEKernel *ensureMoEKernel() const;

        mutable int grouped_gateup_desc_table_id_ = -1;
        mutable int grouped_gateup_desc_table_num_experts_ = 0;
        mutable int grouped_gateup_desc_table_d_model_ = 0;
        mutable int grouped_gateup_desc_table_intermediate_ = 0;

        mutable int grouped_down_desc_table_id_ = -1;
        mutable int grouped_down_desc_table_num_experts_ = 0;
        mutable int grouped_down_desc_table_d_model_ = 0;
        mutable int grouped_down_desc_table_intermediate_ = 0;

        DeviceMoELayerRuntime *moe_runtime_layer_ = nullptr;
        bool moe_runtime_table_initialized_ = false;
        bool moe_prefill_runtime_grouping_available_ = false;
        bool moe_prefill_fixed_topology_available_ = false;
    };

    /**
     * @brief Shared expert FFN stage: always-active dense SwiGLU
     *
     * Runs standard SwiGLU (gate_proj → up_proj → silu(gate)*up → down_proj)
     * using the shared expert weights. Executes for ALL tokens unconditionally.
     */
    class SharedExpertFFNStage : public IComputeStage, public IWorkspaceConsumer
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            TensorBase *input = nullptr;  ///< Normalized hidden [seq_len, d_model]
            TensorBase *gate_w = nullptr; ///< Shared expert gate [intermediate, d_model]
            TensorBase *up_w = nullptr;   ///< Shared expert up [intermediate, d_model]
            TensorBase *down_w = nullptr; ///< Shared expert down [d_model, intermediate]
            TensorBase *output = nullptr; ///< Output [seq_len, d_model]
            int seq_len = 0;
            int d_model = 0;
            int intermediate = 0;

            BufferId input_buffer_id = BufferId::NORMALIZED;
            BufferId output_buffer_id = BufferId::MOE_SHARED_EXPERT_OUTPUT;

            // =================================================================
            // Phase 7: PreparedWeightRef for direct kernel resolution
            // =================================================================
            std::optional<PreparedWeightRef> prepared_ref_gate;
            std::optional<PreparedWeightRef> prepared_ref_up;
            std::optional<PreparedWeightRef> prepared_ref_down;
            PreparedWeightStore *prepared_store = nullptr;
        };

        explicit SharedExpertFFNStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        bool validatePreparedWeights(std::string *error) const override;
        ComputeStageType type() const override { return ComputeStageType::MOE_SHARED_EXPERT_FFN; }
        std::string name() const override { return "shared_expert_ffn"; }
        size_t estimatedFlops() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override;
        bool supportsWarmupDependentGraphCapture() const override;
        bool supportsPaddedPrefillGraphCapturePreflight() const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

        // =====================================================================
        // IWorkspaceConsumer Implementation
        // =====================================================================
        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;
        void unbindWorkspace() override;
        bool hasWorkspace() const override;
        DeviceWorkspaceManager *getWorkspace() const override;

    private:
        Params params_;
        DeviceWorkspaceManager *bound_workspace_ = nullptr; ///< Workspace for shared expert GEMM engines

        mutable ITensorGemm *cached_gate_gemm_ = nullptr;
        mutable ITensorGemm *cached_up_gemm_ = nullptr;
        mutable ITensorGemm *cached_down_gemm_ = nullptr;

        mutable std::vector<bool> shared_expert_mask_;
        mutable std::vector<std::shared_ptr<TensorBase>> shared_gate_views_;
        mutable std::vector<std::shared_ptr<TensorBase>> shared_up_views_;
        mutable std::vector<std::shared_ptr<TensorBase>> shared_down_views_;
        mutable std::vector<ITensorGemm *> shared_prepared_gate_gemm_;
        mutable std::vector<ITensorGemm *> shared_prepared_up_gemm_;
        mutable std::vector<ITensorGemm *> shared_prepared_down_gemm_;
        mutable std::vector<std::shared_ptr<ITensorGemm>> shared_owned_kernels_;
        mutable std::shared_ptr<void> shared_packed_gate_lifetime_;
        mutable std::shared_ptr<void> shared_packed_up_lifetime_;
        mutable std::shared_ptr<void> shared_packed_down_lifetime_;

        mutable std::shared_ptr<FP32Tensor> scratch_gate_;
        mutable std::shared_ptr<FP32Tensor> scratch_up_;
        mutable int scratch_seq_len_ = 0;

        void ensureGemmEnginesCached() const;
        bool ensureSharedGroupedGateUpDescriptorTable(IMoEKernel *kernel, int d_model, int intermediate) const;
        bool ensureSharedGroupedDownDescriptorTable(IMoEKernel *kernel, int d_model, int intermediate) const;
        bool tryGroupedVerifierPrefill(IMoEKernel *kernel, int d_model, int intermediate) const;
        bool tryGroupedDecode(IMoEKernel *kernel, int d_model, int intermediate) const;

        mutable int shared_grouped_gateup_desc_table_id_ = -1;
        mutable int shared_grouped_gateup_desc_table_d_model_ = 0;
        mutable int shared_grouped_gateup_desc_table_intermediate_ = 0;

        mutable int shared_grouped_down_desc_table_id_ = -1;
        mutable int shared_grouped_down_desc_table_d_model_ = 0;
        mutable int shared_grouped_down_desc_table_intermediate_ = 0;

        /// Cached MoE kernel for SwiGLU fallback
        mutable IMoEKernel *moe_kernel_ = nullptr;
        IMoEKernel *ensureMoEKernel() const;

    public:
        // Test accessors
        void setMoEKernelForTesting(IMoEKernel *kernel) { moe_kernel_ = kernel; }
        void setScratchSeqLenForTesting(int n) { scratch_seq_len_ = n; }
    };

    /**
     * @brief Shared expert sigmoid gate stage
     *
     * Computes: output = sigmoid(gate_inp · input) ⊙ shared_expert_output
     * - gate_inp: [d_model] vector
     * - input: [seq_len, d_model]
     * - shared_expert_output: [seq_len, d_model] (modified in-place)
     */
    class SharedExpertGateStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            TensorBase *input = nullptr;         ///< Normalized hidden [seq_len, d_model]
            TensorBase *gate_inp = nullptr;      ///< Gate vector [d_model]
            TensorBase *shared_output = nullptr; ///< Shared expert output (in-place) [seq_len, d_model]
            int seq_len = 0;
            int d_model = 0;

            BufferId input_buffer_id = BufferId::NORMALIZED;
            BufferId output_buffer_id = BufferId::MOE_SHARED_EXPERT_OUTPUT;
        };

        explicit SharedExpertGateStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_SHARED_EXPERT_GATE; }
        std::string name() const override { return "shared_expert_gate"; }
        bool allowsZeroOutput() const override
        {
            // The sigmoid gate can saturate to exactly zero for very negative
            // gate dot-products. That is a valid model result, not an
            // uninitialized shared-expert buffer.
            return true;
        }
        size_t estimatedFlops() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override;
        bool supportsWarmupDependentGraphCapture() const override;
        bool supportsPaddedPrefillGraphCapturePreflight() const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

    private:
        Params params_;

        /// Cached MoE kernel for sigmoid gating
        mutable IMoEKernel *moe_kernel_ = nullptr;
        IMoEKernel *ensureMoEKernel() const;

    public:
        // Test accessors
        void setMoEKernelForTesting(IMoEKernel *kernel) { moe_kernel_ = kernel; }
    };

} // namespace llaminar2
