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
            bool force_grouped_verifier_prefill_for_decode = false;
            bool force_decode_equivalent_verifier_prefill = false;

            /**
             * @brief Require GPU decode to consume routing tensors on device.
             *
             * Decode-equivalent verifier row replay gathers one row from the
             * all-position routing tensors.  For GPU backends those row tensors
             * are DEVICE_AUTHORITATIVE, so the scoped one-token expert replay
             * must use grouped `FromRouting` kernels instead of falling back to
             * `data()` on a stale host mirror.  When this flag is true, failure
             * to use the device route tensors is a correctness error.
             */
            bool require_device_routing_tensor_decode = false;

            /**
             * @brief Own both routed and shared verifier branches in one stage.
             *
             * The promoted verifier fast path is a safe composite, not the old
             * single-table routed+shared shortcut.  It runs the routed experts
             * through the proven grouped verifier-prefill path, runs the shared
             * expert through decode-equivalent M=2..4 GEMV hooks, then applies
             * the normal shared sigmoid gate plus routed residual add.  The graph
             * must not add separate shared-expert FFN/gate nodes for the same
             * layer when this is enabled.
             */
            bool combine_shared_expert_in_verifier = false;
            TensorBase *shared_gate_w = nullptr;
            TensorBase *shared_up_w = nullptr;
            TensorBase *shared_down_w = nullptr;
            TensorBase *shared_gate_inp = nullptr;
            std::optional<PreparedWeightRef> prepared_shared_ref_gate;
            std::optional<PreparedWeightRef> prepared_shared_ref_up;
            std::optional<PreparedWeightRef> prepared_shared_ref_down;

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
        /**
         * @brief Drop per-request fused decode warmup state.
         *
         * clear_cache() may reset backend pointer-table readiness while keeping
         * this graph object alive, so the next request must warm routed MoE
         * decode again before capture is allowed.
         */
        void resetSessionState() override
        {
            IComputeStage::resetSessionState();
            runtime_grouped_decode_warmed_ = false;
        }

        /**
         * @brief Clear stream ownership without invalidating captured decode tables.
         *
         * The runtime grouped decode path warms descriptor and pointer-table
         * slots before graph capture. A preserved CUDA/HIP graph executable
         * still reads those device slots by address, so request-boundary replay
         * preservation must not flip runtime_grouped_decode_warmed_ back to
         * false. True topology changes still use resetSessionState(),
         * invalidate(), or graph rebuild paths.
         */
        void resetSessionStatePreservingCapturedReplay() override
        {
            IComputeStage::resetSessionState();
        }
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
        void setRuntimeGroupedDecodeWarmedForTesting(bool warmed) { runtime_grouped_decode_warmed_ = warmed; }
        bool usesCPUDecodeEquivalentVerifierPrefillForTesting() const
        {
            return params_.force_decode_equivalent_verifier_prefill;
        }
        /**
         * @brief Test-only predicate for the fixed-topology M=1 verifier replay.
         *
         * LocalTP and overlay runners may own only a subset of experts.  Those
         * lanes must not claim the full-ownership descriptor-table replay path;
         * they use the mask-aware device grouped route instead.
         */
        bool usesFixedTopologyGroupedVerifierReplayForTesting() const
        {
            return params_.force_grouped_verifier_prefill_for_decode &&
                   params_.seq_len == 1 &&
                   canUseFixedTopologyGroupedPrefill();
        }
        TensorBase *combinedSharedGateInputForTesting() const
        {
            return effectiveSafeCompositeSharedGateInput();
        }

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
        /**
         * @brief Output scratch for verifier rows replayed through decode.
         *
         * executeSingleToken() uses scratch_out_ as an internal SwiGLU/down
         * temporary.  The per-row verifier output must not alias that scratch,
         * otherwise the helper can accumulate into its own intermediate buffer.
         */
        mutable std::shared_ptr<FP32Tensor> verifier_output_row_;
        /**
         * @brief Device-owned routing row scratch for verifier replay.
         *
         * MoERoutingStage writes all-position verifier routing tensors on the
         * GPU.  The expert stage must gather the selected row from those
         * device-owned tensors instead of reading stale host mirrors; otherwise
         * CUDA/ROCm can replay a different expert set than the routing stage
         * actually produced.
         */
        mutable std::shared_ptr<FP32Tensor> verifier_routing_indices_row_;
        mutable std::shared_ptr<FP32Tensor> verifier_routing_weights_row_;
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

        /**
         * @brief True after routed single-token fused decode has staged runtime pointer arrays.
         *
         * CUDA and ROCm fused MoE decode read device-side pointer tables for the
         * gate/up and down scratch slots during graph replay.  Those tables are
         * populated by an eager warmup run because graph capture must not contain
         * the host-to-device metadata uploads.  Keep this flag tied to the current
         * workspace, descriptor tables, and runtime layer so capture can only
         * start after that exact route has succeeded.
         */
        mutable bool runtime_grouped_decode_warmed_ = false;

        /// Fast path for decode (seq_len=1): avoids token grouping, gather/scatter,
        /// and per-expert heap allocations. Uses routing results directly.
        bool executeSingleToken(IDeviceContext *ctx);

        /**
         * @brief Execute verifier-sized batches as a series of decode rows.
         *
         * MTP state publication restores live KV/GDN/conv state from a selected
         * verifier row.  That only remains sound when the verifier row was
         * produced by the same math as ordinary one-token decode.  This helper
         * enforces that contract for CPU, CUDA, and ROCm by running M=1 expert
         * projections per row, while still using backend tensor kernels for the
         * heavy GPU work.
         */
        bool executeDecodeEquivalentVerifierPrefill(IDeviceContext *ctx);

        void ensureGemmEnginesCached();
        bool ensureGemmEnginesForExperts(const std::vector<int> &expert_ids);
        bool ensureGroupedGateUpDescriptorTable(IMoEKernel *kernel, int d_model, int intermediate);
        bool ensureGroupedDownDescriptorTable(IMoEKernel *kernel, int d_model, int intermediate);
        bool ensureCombinedSharedVerifierResources(IMoEKernel *kernel, int d_model, int intermediate);
        bool initializeMoERuntimeTableForGroupedDecode();
        bool initializeMoERuntimeTableForGroupedPrefill();
        bool initializeFixedTopologyGroupedPrefill();
        bool runtimeTableHasActiveGroupedDecodeBank() const;
        bool canUseRuntimePrefillGrouping() const;
        bool canUseFixedTopologyGroupedPrefill() const;
        /**
         * @brief True when verifier rows can use the safe routed+shared composite path.
         *
         * The rejected shortcut treated the shared expert as an extra routed expert
         * in one MoE prefill table.  This predicate guards the replacement design:
         * routed experts use the proven grouped verifier pipeline, shared expert
         * rows use decode-equivalent GEMV-many projections, and the stage combines
         * those two branch outputs with the normal shared-gate add kernel.
         */
        bool canUseSafeCombinedSharedVerifierComposite() const;
        TensorBase *effectiveSafeCompositeSharedGateInput() const;
        bool executeSafeCombinedSharedVerifierComposite(IMoEKernel *kernel) const;
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

        mutable int combined_shared_desc_table_d_model_ = 0;
        mutable int combined_shared_desc_table_intermediate_ = 0;
        mutable std::shared_ptr<FP32Tensor> combined_shared_gate_inp_fp32_;
        mutable TensorBase *combined_shared_gate_inp_source_ = nullptr;
        mutable ITensorGemm *combined_shared_gate_gemm_ = nullptr;
        mutable ITensorGemm *combined_shared_up_gemm_ = nullptr;
        mutable ITensorGemm *combined_shared_down_gemm_ = nullptr;
        mutable std::shared_ptr<FP32Tensor> combined_routed_output_;
        mutable std::shared_ptr<FP32Tensor> combined_shared_output_;
        mutable std::shared_ptr<FP32Tensor> combined_shared_gate_scratch_;
        mutable std::shared_ptr<FP32Tensor> combined_shared_up_scratch_;

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
            bool force_grouped_verifier_prefill_for_decode = false;
            bool force_decode_equivalent_verifier_prefill = false;
            /**
             * @brief Bypass normal grouped decode shortcuts for verifier replay.
             *
             * Decode-equivalent verifier publication needs a stable one-row
             * source of truth.  GPU grouped shared-expert decode may use
             * split-K/concurrent kernels that are valid for normal inference but
             * are not the canonical rowwise verifier oracle.
             */
            bool disable_grouped_decode_shortcut = false;

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
        /**
         * @brief Drop per-request grouped decode warmup state.
         *
         * The backend owns graph-replayed pointer arrays for shared-expert
         * decode. Session reset may clear those arrays without rebuilding the
         * stage, so capture must require a fresh warmup.
         */
        void resetSessionState() override
        {
            IComputeStage::resetSessionState();
            grouped_decode_warmed_ = false;
        }

        /**
         * @brief Preserve grouped shared-expert pointer tables for graph replay.
         *
         * Normal request reset marks grouped decode cold so a following capture
         * gets a warmup pass. If the caller is deliberately keeping the already
         * captured executable alive, those warmed tables are part of the
         * executable contract and must remain valid.
         */
        void resetSessionStatePreservingCapturedReplay() override
        {
            IComputeStage::resetSessionState();
        }
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        bool usesGroupedVerifierPrefillRouteForTesting() const;
        bool usesCPUDecodeEquivalentVerifierPrefillForTesting() const;
        bool usesGroupedDecodeForTesting() const;

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
        mutable std::shared_ptr<FP32Tensor> scratch_input_row_;
        mutable std::shared_ptr<FP32Tensor> scratch_output_row_;
        mutable int scratch_seq_len_ = 0;
        /**
         * @brief True once normal single-token grouped decode has populated
         * graph-owned runtime pointer arrays for the currently bound workspace.
         *
         * The CUDA grouped decode kernels read device-side arrays of scratch
         * tensor pointers during graph replay.  Those arrays are produced by a
         * warmup execution outside capture; capture is only safe after that
         * exact route has succeeded with the current scratch/workspace owner.
         */
        mutable bool grouped_decode_warmed_ = false;

        void ensureGemmEnginesCached() const;
        bool ensureSharedGroupedGateUpDescriptorTable(IMoEKernel *kernel, int d_model, int intermediate) const;
        bool ensureSharedGroupedDownDescriptorTable(IMoEKernel *kernel, int d_model, int intermediate) const;
        bool shouldUseGroupedVerifierPrefillRoute() const;
        bool shouldUseDecodeEquivalentVerifierPrefill() const;
        bool shouldUseGroupedDecodeRoute() const;
        bool tryGroupedVerifierPrefill(IMoEKernel *kernel, int d_model, int intermediate) const;
        bool tryGroupedDecode(IMoEKernel *kernel, int d_model, int intermediate) const;

        /**
         * @brief Execute shared expert verifier rows through the decode path.
         *
         * The shared expert has no routing state, but its quantized GEMM kernels
         * can still choose different M=2..4 math than M=1 decode.  State
         * publication needs the canonical M=1 GEMM/SwiGLU/down path so the next
         * accepted token continues exactly like serial decode.
         */
        bool executeDecodeEquivalentVerifierPrefill(
            IDeviceContext *ctx, IMoEKernel *kernel,
            int d_model, int intermediate);

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
        void setGroupedDecodeWarmedForTesting(bool warmed) { grouped_decode_warmed_ = warmed; }
    };

    /**
     * @brief Shared expert sigmoid gate stage
     *
     * Computes one of two equivalent shared-expert epilogues:
     * - In-place gate: shared_output *= sigmoid(gate_inp · input)
     * - Fused combine: combined_output = routed_residual + shared_output * sigmoid(...)
     *
     * The fused-combine form is used by single-device MoE graphs to avoid a
     * separate ResidualAddStage between the shared expert and routed expert paths.
     * - gate_inp: [d_model] vector
     * - input: [seq_len, d_model]
     * - shared_expert_output: [seq_len, d_model]
     */
    class SharedExpertGateStage : public IComputeStage, public IWorkspaceConsumer
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            TensorBase *input = nullptr;         ///< Normalized hidden [seq_len, d_model]
            TensorBase *gate_inp = nullptr;      ///< Gate vector [d_model]
            TensorBase *shared_output = nullptr; ///< Shared expert output [seq_len, d_model]

            /// Optional routed MoE output to add after shared-expert gating.
            /// When both routed_residual and combined_output are set, the stage
            /// writes the gated shared contribution back to shared_output and
            /// writes routed_residual + shared_output to combined_output in the
            /// same kernel. When omitted, the legacy in-place gate path is used.
            TensorBase *routed_residual = nullptr;
            TensorBase *combined_output = nullptr;
            int seq_len = 0;
            int d_model = 0;

            BufferId input_buffer_id = BufferId::NORMALIZED;
            BufferId output_buffer_id = BufferId::MOE_SHARED_EXPERT_OUTPUT;
            BufferId residual_buffer_id = BufferId::MOE_COMBINED_OUTPUT;
            BufferId combined_output_buffer_id = BufferId::ATTN_PROJ;
        };

        explicit SharedExpertGateStage(Params params);
        ~SharedExpertGateStage() override;

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
        bool supportsPaddedPrefillRealLengthContract() const override;
        bool hasPrefillReplayParams() const override { return params_.device_id.is_gpu() && params_.seq_len > 1; }
        void updatePrefillReplayParams(const PrefillReplayParams &replay) override;
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;
        bool needsGraphLaunchPreparation() const override { return hasPrefillReplayParams(); }
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;
        void unbindWorkspace() override;
        bool hasWorkspace() const override;
        DeviceWorkspaceManager *getWorkspace() const override;

        /**
         * @brief Clear request-local padded prefill parameters while preserving
         * kernel handles and warmed model weights.
         */
        void resetSessionState() override;
        /**
         * @brief Preserve shared-expert effective-length storage for graph replay.
         *
         * The standalone shared-expert verifier path uses a workspace-backed
         * scalar that is refreshed before every padded prefill replay. Request
         * reset may clear host mirrors, but it must not make the preserved
         * executable lose the stable scalar identity.
         */
        void resetSessionStatePreservingCapturedReplay() override;
        /**
         * @brief Preserve warmed shared-expert metadata for capture-from-Initialized.
         */
        void resetSessionStatePreservingLazyInitialization() override;

    private:
        struct GpuEffectiveSeqLenState;

        Params params_;

        mutable std::shared_ptr<FP32Tensor> fp32_gate_inp_;
        mutable TensorBase *fp32_gate_source_ = nullptr;
        TensorBase *effectiveGateInput() const;
        bool gateInputReadyForGraphCapture() const;

        /// Cached MoE kernel for sigmoid gating
        mutable IMoEKernel *moe_kernel_ = nullptr;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;
        int prefill_effective_seq_len_ = 0;
        bool prefill_replay_params_set_ = false;
        std::unique_ptr<GpuEffectiveSeqLenState> gpu_effective_seq_len_state_;
        IMoEKernel *ensureMoEKernel() const;
        int effectivePrefillSeqLen() const;
        void refreshPinnedEffectiveSeqLen();
        bool ensureGpuEffectiveSeqLenStateInitialized();
        bool uploadGpuEffectiveSeqLen();
        void releaseGpuEffectiveSeqLenState();

    public:
        // Test accessors
        void setMoEKernelForTesting(IMoEKernel *kernel) { moe_kernel_ = kernel; }
    };

} // namespace llaminar2
