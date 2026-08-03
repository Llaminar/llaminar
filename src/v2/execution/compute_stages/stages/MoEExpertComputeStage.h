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
#include "../../config/RuntimeConfig.h"
#include "../../moe/ExpertWeightTransfer.h"
#include "../../moe/MoERebalanceController.h"
#include "../../moe/MoEExpertWeightService.h"
#include "../../moe/MoERuntimeTable.h"
#include "../../moe/DeviceMoERebalanceController.h"

#include <memory>
#include <cstdint>
#include <string>
#include <stdexcept>
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
    class GpuExpertSlotPool;
    class GpuExpertTransferStagingPool;
    class ILocalTPContext;
    class DeviceMoERebalanceTransferState;

    /**
     * @brief Select how a grouped LLEP invocation assigns the current batch.
     *
     * This policy is deliberately independent of the physical transport mode.
     * A stage can own compact NCCL/RCCL transport resources because a prefix
     * restore must rehydrate persistent expert payloads while still assigning
     * the current grouped verifier batch exclusively across experts that are
     * already resident.
     *
     * Grouped verifier rows always use
     * @ref LogicalPositionResidentOnly. Moving an expert payload for a
     * handful of speculative rows costs far more than executing those rows on
     * an existing owner or replica, and it would place a payload collective in
     * every MoE layer of every verifier replay. Long prefill may select
     * @ref TransferBackedCurrentBatch after its routed-row economy gate has
     * passed.
     */
    enum class PrefillLLEPAssignmentMode : uint8_t
    {
        /**
         * Split rows only across the active bank's resident participant mask.
         *
         * This mode never creates or consumes a current-batch weight-transfer
         * plan. Prefix-runtime rehydration, when requested, is a separate
         * transaction that completes before this assignment begins.
         */
        LogicalPositionResidentOnly = 0,

        /**
         * Plan missing arrivals, execute compact payload collectives, publish
         * the resulting runtime bank, and then apply the assignment spans.
         *
         * Graph construction may select this only for an amortizable long
         * prefill in a homogeneous graph-capturable LocalTP domain.
         */
        TransferBackedCurrentBatch = 1,
    };

    /**
     * @brief Identify why a captured LLEP payload transaction is executing.
     *
     * Current-batch movement consumes the routing evidence that selected the
     * new placement and therefore starts a new histogram window after apply.
     * Prefix-runtime rehydration has different semantics: it reconstructs
     * payload bytes for placement and evidence already restored from a
     * portable prefix snapshot. Erasing that evidence would make the first
     * maintenance decision after a cache hit differ from uninterrupted
     * execution.
     *
     * The purpose is a required method argument rather than inferred from
     * pointer identity or mutable runtime state. This keeps graph construction
     * declarative and makes an ambiguous payload publication unrepresentable.
     */
    enum class PrefillLLEPTransferPurpose : uint8_t
    {
        CurrentBatchMovement = 0,
        PrefixRuntimeRehydration = 1,
    };

    /**
     * @brief Derive the immutable apply policy for one LLEP transaction.
     *
     * @param base_config Graph-owned rebalance policy shared by both captured
     *                    transaction shapes.
     * @param purpose     Semantic owner of this transaction.
     * @return A value copy whose histogram lifecycle matches @p purpose.
     */
    [[nodiscard]] inline DeviceMoERebalanceConfig
    prefillLLEPTransferConfig(
        const DeviceMoERebalanceConfig &base_config,
        PrefillLLEPTransferPurpose purpose)
    {
        DeviceMoERebalanceConfig result = base_config;
        const uint32_t reset_histograms =
            static_cast<uint32_t>(
                DeviceMoERebalanceFlags::ResetHistogramsAfterApply);
        switch (purpose)
        {
        case PrefillLLEPTransferPurpose::CurrentBatchMovement:
            result.flags |= reset_histograms;
            break;
        case PrefillLLEPTransferPurpose::PrefixRuntimeRehydration:
            result.flags &= ~reset_histograms;
            break;
        default:
            throw std::invalid_argument(
                "Unknown prefill LLEP transfer purpose");
        }
        return result;
    }

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
            /**
             * @brief Graph-local kernel owner shared only with this stage's router.
             *
             * The router publishes Q8 hidden rows and route metadata into this
             * device-resident pipeline context. Main decode, MTP sidecars, and
             * rebalance maintenance each receive different owners.
             */
            std::shared_ptr<MoERoutedPipelineKernelOwner> routed_pipeline_kernel_owner;

            // Expert weights (3D packed tensors) — used by CPU path
            TensorBase *gate_exps = nullptr; ///< [num_experts, intermediate, d_model]
            TensorBase *up_exps = nullptr;   ///< [num_experts, intermediate, d_model]
            TensorBase *down_exps = nullptr; ///< [num_experts, d_model, intermediate]
            int expert_intermediate = 0;

            // Whole-expert-ID apportionment across participants.
            // When active, this rank only computes experts in
            // [local_expert_start, local_expert_start + local_expert_count).
            // Set by graph builder when TP degree > 1.
            // -1 means all expert IDs are local (replicated or single-device mode).
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
            /// to participants per-token to balance load. Resident participants
            /// have GEMM engines for replicated experts; only one computes each
            /// per token.
            ExpertReplicaSet replica_set;

            /// This rank's participant ID (for per-token replica dispatch).
            int my_socket_id = 0;

            /// Number of participants in the routed expert domain. Zero means
            /// infer from replica metadata for legacy single-device paths.
            int participant_count = 0;

            /// Policy for assigning already-selected routed expert rows to
            /// domain participants. StaticOwner follows the placement owner;
            /// LLEP preserves router top-k choices and balances routed
            /// row spans across resident owners/replicas through runtime
            /// prefill grouping.
            RoutedExpertAssignmentPolicy routed_assignment_policy =
                RoutedExpertAssignmentPolicy::StaticOwner;

            /**
             * @brief Immutable graph-lowered contract for routed row ownership.
             *
             * ParticipantAssigned applies the configured resident scheduling
             * policy and publishes a participant-local contribution. In
             * FullyReplicatedLocal mode every participant owns every complete
             * expert and executes every selected row locally; resident
             * assignment and canonical route contribution buffers are then
             * forbidden because either would reintroduce a tiny decode
             * collective or sum duplicate complete outputs.
             */
            RoutedExpertRowExecutionPolicy routed_row_execution_policy =
                RoutedExpertRowExecutionPolicy::ParticipantAssigned;

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
            std::shared_ptr<GpuExpertSlotPool> gpu_direct_slot_pool;

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
             * @brief Retain this main-verifier layer's routes for later commit.
             *
             * Sidecar and main verifier stages both use grouped kernels, but
             * only the main target graph publishes accepted rows into decode
             * maintenance history. Keeping this policy explicit prevents a
             * sidecar from accidentally advertising a deferred publication
             * transaction merely because it uses grouped execution.
             */
            bool defer_grouped_verifier_histogram_publication = false;

            /**
             * @brief Device-owned absolute position for every grouped row.
             *
             * Batch-invariant resident LLEP uses this row to rotate equal-load
             * participant ties without consulting speculative workload
             * history. GPU verifier graphs bind the same persistent position
             * row consumed by RoPE; a null pointer is fatal when resident
             * assignment is selected.
             */
            const int32_t *absolute_position_ids_device = nullptr;

            /**
             * @brief Device-owned logical row count inside the physical graph width.
             *
             * Routing, resident LLEP assignment, grouped expert execution, and
             * the shared epilogue form one row-geometry transaction. Variable-M
             * GPU graphs bind the same stable request scalar to every member so
             * padded suffix rows can never be interpreted as routed work by one
             * stage after another stage masked them. No host mirror or replay
             * upload may participate in this contract.
             */
            const int32_t *active_row_count_device = nullptr;

            /**
             * @brief Require GPU decode to consume routing tensors on device.
             *
             * Some verifier and overlay M=1 lanes bind routing tensors directly
             * instead of consuming DeviceMoELayerRuntime state.  For GPU
             * backends those tensors can be DEVICE_AUTHORITATIVE, so decode
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
             * expert through decode-equivalent M=1..4 GEMV hooks, then applies
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

            /**
             * @brief Optional ownership-invariant LocalTP publication target.
             *
             * GPU fused decode and grouped-prefill kernels write one weighted
             * row per original router slot into `[seq_len, top_k, d_model]`.
             * They do not collapse participant-local routes into @ref output;
             * a following FP32 collective and canonical reducer own that sum.
             */
            TensorBase *canonical_route_contributions = nullptr;

            // Buffer IDs for coherence
            BufferId input_buffer_id = BufferId::NORMALIZED;
            BufferId output_buffer_id = BufferId::MOE_COMBINED_OUTPUT;
            BufferId canonical_route_contributions_buffer_id =
                BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS;
            bool output_registered_in_arena = true;

            // =================================================================
            // Phase 7: PreparedWeightStore for decode-time fallback resolution
            // =================================================================
            PreparedWeightStore *prepared_store = nullptr;

            // Stable graph-facing MoE runtime placement state. Owned by the graph/model
            // layer; stages only cache the per-layer device pointer.
            IMoERuntimeTable *moe_runtime_table = nullptr;
            bool use_runtime_prefill_grouping = false;

            /// LocalTP context for future full LLEP current-batch row exchange.
            /// Required by graph-capturable homogeneous GPU LLEP prefill once
            /// assignment spans are promoted from planning to execution.
            ILocalTPContext *prefill_llep_tp_ctx = nullptr;

            /// Optional graph-owned transfer backing for full LLEP prefill.
            /// These handles let current-batch assignment spans import missing
            /// expert descriptors on device before route_participant_ids are
            /// rewritten. The expert stage borrows this memory; it never owns
            /// or allocates transfer slots.
            DeviceMoEExpertDirectoryEntry *prefill_llep_transfer_slots = nullptr;
            uint32_t prefill_llep_transfer_slot_count = 0;
            uint64_t prefill_llep_payload_slot_bytes = 0;
            uint32_t prefill_llep_payload_slot_capacity = 0;
            DeviceMoERebalanceTransferMode prefill_llep_transfer_mode =
                DeviceMoERebalanceTransferMode::ResidentOnly;
            /**
             * @brief Current-batch LLEP assignment policy fixed at graph build.
             *
             * Do not infer this policy from whether transport pointers happen
             * to be bound. Prefix restore legitimately binds those pointers to
             * a resident-only verifier stage, so pointer presence is not proof
             * that current-batch expert migration is legal.
             */
            PrefillLLEPAssignmentMode prefill_llep_assignment_mode =
                PrefillLLEPAssignmentMode::LogicalPositionResidentOnly;
            /**
             * @brief Prepend exact prefix-placement payload reconstruction.
             *
             * This immutable graph-build flag selects a one-shot captured
             * transaction. The runtime table already owns the desired transfer
             * list in persistent device scratch; the stage executes that list
             * before decode or prefill routing can observe placement.
             */
            bool prefix_runtime_device_rehydration = false;
            DeviceMoERebalanceConfig prefill_llep_rebalance_config;
            std::shared_ptr<DeviceMoERebalanceTransferState> prefill_llep_transfer_state;
            /**
             * @brief Event identity dedicated to restore-time payload movement.
             *
             * A restored suffix can execute rehydration and a new current-batch
             * LLEP movement in the same layer graph. They share one serialized
             * transfer stream/workspace lane, but must not alias event objects:
             * two records of one event inside one captured graph make dependency
             * ownership ambiguous on CUDA and HIP.
             */
            std::shared_ptr<DeviceMoERebalanceTransferState>
                prefix_runtime_rehydration_transfer_state;
            std::string prefill_llep_workspace_name;

            /*
             * Runtime decode always consumes runtime top-k ids/weights. This
             * flag controls where the expert weight descriptors come from:
             * false keeps static/off decode on the fast immutable descriptor
             * tables; true makes graph replay observe mutable runtime placement
             * descriptors after device-side rebalance applies ownership changes.
             */
            bool runtime_decode_uses_mutable_descriptors = false;

            /*
             * True when graph construction has published an explicit
             * multi-participant owner/residency bank for masked decode.  This is
             * independent of descriptor mutability: resident-only/static decode
             * can still use immutable descriptor tables while relying on the
             * runtime bank for correct apportioned-expert ownership metadata.
             */
            bool runtime_decode_has_explicit_owner_metadata = false;

            // Phase C: Cached slab refs for store-based resolution and rebalance
            std::optional<ExpertSlabRef> gate_slab_ref;
            std::optional<ExpertSlabRef> up_slab_ref;
            std::optional<ExpertSlabRef> down_slab_ref;
        };

        explicit MoEExpertComputeStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        bool validatePreparedWeights(std::string *error) const override;
        ComputeStageType type() const override { return ComputeStageType::MOE_EXPERT_FFN; }
        /**
         * @brief Report transfer-backed LLEP collectives embedded in this stage.
         *
         * Ordinary grouped expert compute is participant-local and is followed
         * by an explicit graph collective. Full LLEP prefill is different: its
         * device-resident movement transaction gathers plans, headers, and
         * payloads inside this stage. The instance-level contract keeps capture
         * planning accurate without misclassifying every MoE expert stage.
         */
        bool isCollectiveStage() const override;
        std::string name() const override { return "moe_ffn"; }
        size_t estimatedFlops() const override;

        /// Layer index this stage belongs to (-1 if unset).
        int layerIndex() const { return params_.layer_idx; }

        /// True for the graph-stable GPU decode path whose placement is
        /// published by mutating persistent runtime descriptor tables.
        bool usesGraphStableRuntimeDecodePlacement() const;

        /**
         * @brief Return whether graph preparation owns this grouped-prefill route.
         *
         * Both ordinary prefill and forced grouped-verifier execution use the
         * same persistent MoE kernel, GEMM descriptor tables, expert mask, and
         * optional runtime-grouping workspace.  Keeping the verifier flag out
         * of this ownership decision ensures every supported verifier depth is
         * prepared before capture instead of entering capture with a cold
         * kernel or unbound descriptor table.
         *
         * @return `true` when `prepareGraphLaunch()` must publish the stable
         * grouped-prefill placement before graph capture begins.
         */
        bool usesGraphStableFixedTopologyPrefillPlacement() const;

        /// True when this stage can consume dynamic MoE placement without a
        /// graph recapture.
        bool usesGraphStableMoEPlacement() const;

        MoEDecodeDescriptorSource runtimeDecodeDescriptorSourceForTesting() const
        {
            return params_.runtime_decode_uses_mutable_descriptors
                       ? MoEDecodeDescriptorSource::RuntimePlacementTable
                       : MoEDecodeDescriptorSource::StaticDescriptorTable;
        }

        /// Test-only visibility for placement metadata stamped onto rebuilt graphs.
        int replicaCountForTesting() const { return params_.replica_set.num_replicated; }
        int replicaParticipantForTesting() const { return params_.my_socket_id; }
        const std::vector<bool> &expertMaskForTesting() const { return params_.expert_mask; }
        RoutedExpertAssignmentPolicy routedExpertAssignmentPolicyForTesting() const
        {
            return params_.routed_assignment_policy;
        }
        RoutedExpertRowExecutionPolicy routedExpertRowExecutionPolicyForTesting() const
        {
            return params_.routed_row_execution_policy;
        }
        bool usesRuntimePrefillGroupingForTesting() const
        {
            return params_.use_runtime_prefill_grouping;
        }
        bool supportsRequestedRoutedAssignmentPolicyForTesting() const
        {
            return supportsRequestedRoutedAssignmentPolicy();
        }
        bool hasMoERuntimeTableForTesting() const { return params_.moe_runtime_table != nullptr; }
        bool hasPrefillLLEPTPContextForTesting() const { return params_.prefill_llep_tp_ctx != nullptr; }
        bool hasTransferBackedPrefillLLEPForTesting() const { return hasTransferBackedPrefillLLEP(); }
        PrefillLLEPAssignmentMode prefillLLEPAssignmentModeForTesting() const noexcept
        {
            return params_.prefill_llep_assignment_mode;
        }
        const int32_t *absolutePositionIdsDeviceForTesting() const noexcept
        {
            return params_.absolute_position_ids_device;
        }
        const std::string &prefillLLEPWorkspaceNameForTesting() const
        {
            return params_.prefill_llep_workspace_name;
        }
        const DeviceMoERebalanceTransferState *
        prefillLLEPTransferStateForTesting() const
        {
            return params_.prefill_llep_transfer_state.get();
        }

        /// With expert-ID apportionment, a participant's output can be all zeros
        /// when no selected experts fall in its local range. The downstream
        /// AllReduce combines partial results across ranks.
        bool allowsZeroOutput() const override
        {
            return params_.local_expert_count >= 0 || !params_.expert_mask.empty();
        }

        /// Update expert mask for dynamic rebalancing (runtime, no rebuild needed).
        /// mask.size() must == num_experts. Returns false on size mismatch.
        bool updateExpertMask(const std::vector<bool> &mask);

        /**
         * @brief Publish replica placement for per-token dynamic dispatch.
         *
         * The expert mask and replica set are two views of one placement
         * transaction. Every owner and every advertised replica must already
         * have a prepared local GEMM engine before the replica set becomes
         * visible to decode. Validating that invariant here turns an incoherent
         * publication into an immediate, attributable failure instead of a
         * later null-engine abort inside the token hot path.
         *
         * @param replicas Authoritative owner and replica placement.
         * @param socket_id Domain-local participant represented by this stage.
         * @throws std::runtime_error if the local expert mask does not contain
         *         an owner or replica advertised as resident on this participant.
         */
        void setReplicaSet(const ExpertReplicaSet &replicas, int socket_id)
        {
            ExpertReplicaSet normalized_replicas = replicas;
            normalized_replicas.rebuildAggregateReplicaFlags();

            /*
             * All participants receive the same replica metadata, while each
             * stage carries only its participant-local residency mask. Check
             * only the forward implication here: every advertised local owner
             * or replica must be resident. A mask may intentionally contain
             * additional cache residents that are not eligible for assignment.
             */
            for (size_t expert = 0; expert < normalized_replicas.owner_socket.size(); ++expert)
            {
                const bool is_owner =
                    normalized_replicas.owner_socket[expert] == socket_id;
                const bool is_replica =
                    normalized_replicas.hasReplicaOnParticipant(
                        params_.layer_idx,
                        static_cast<int>(expert),
                        socket_id);
                if ((!is_owner && !is_replica) ||
                    (expert < params_.expert_mask.size() && params_.expert_mask[expert]))
                {
                    continue;
                }

                throw std::runtime_error(
                    "MoE replica publication advertised expert " +
                    std::to_string(expert) + " as resident on participant " +
                    std::to_string(socket_id) + " for layer " +
                    std::to_string(params_.layer_idx) +
                    " before the expert mask and GEMM engines were published");
            }

            params_.replica_set = std::move(normalized_replicas);
            params_.my_socket_id = socket_id;
            // Pre-build prefill mask: single-lookup replaces multi-branch check
            if (params_.replica_set.num_replicated > 0 && !params_.expert_mask.empty())
                params_.replica_set.buildPrefillMask(socket_id, params_.expert_mask, params_.layer_idx);
            else
                params_.replica_set.prefill_mask.clear();
            grouped_gateup_desc_table_dirty_ = true;
            grouped_down_desc_table_dirty_ = true;
            moe_runtime_table_initialized_ = false;
            invalidateFixedTopologyMaskPublication();
            if (!refreshGraphStablePlacement(/*preserve_capture_ready=*/true))
            {
                throw std::runtime_error(
                    "MoE expert replica publication failed to refresh graph-stable runtime tables");
            }
        }

        /// Detach and serialize packed weights for a departing expert.
        /// Returns serialized gate/up/down blobs. After this call, the expert's
        /// GEMM engines have empty weights (will be cleaned up in Phase 1 of
        /// updateExpertMaskAndPrepareEngines).
        ExpertWeightBlobs detachAndSerializeExpert(int expert_id);

        /// Serialize packed weights for an expert without detaching.
        /// The owner keeps its GEMM engines intact. Used for replica transfers
        /// where both owner and replica participants need the weights.
        ExpertWeightBlobs serializeExpert(int expert_id) const;

        /// Directly copy same-backend GPU packed expert weights from a sibling
        /// stage into this stage. Returns requested experts that are now resident
        /// on this stage, including experts that were already present before the copy.
        std::vector<int> transferExpertsGPUDirectFrom(
            MoEExpertComputeStage &source,
            const std::vector<int> &expert_ids,
            void *source_producer_stream);

        /// Return requested experts that do not currently have all prepared
        /// gate/up/down GEMM engines on this stage.
        std::vector<int> missingPreparedExpertIds(
            const std::vector<int> &expert_ids) const;

        /// Stage same-backend GPU expert weights from a sibling stage into this
        /// stage's transfer slots without publishing active GEMM engines.
        std::vector<int> stageExpertsGPUDirectToTransferSlotsFrom(
            MoEExpertComputeStage &source,
            const std::vector<int> &expert_ids,
            void *source_producer_stream,
            GpuDirectTransferSlotArrivals *staged_arrivals,
            size_t active_arrival_capacity = 0,
            size_t staging_pool_capacity = 0,
            std::vector<std::shared_ptr<GpuExpertTransferStagingPool>>* transfer_staging_pools = nullptr);

        /// Activate previously staged GPU-direct transfer-slot arrivals into
        /// active expert slots and publish their GEMM engines. Must run on the
        /// runner thread with an explicit destination stream.
        std::vector<int> activateGpuDirectTransferSlotArrivals(
            const GpuDirectTransferSlotArrivals &arrivals,
            void *activation_stream,
            GpuDirectTransferCompletion *completion_out = nullptr,
            bool retain_pending_completion = true);

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
        /**
         * @brief Describe every routed-expert graph-capture readiness invariant.
         *
         * Warmup-dependent capture failures are fatal.  Returning the complete
         * predicate state here lets the capture controller identify whether the
         * missing prerequisite is the backend kernel, descriptor tables, the
         * runtime placement bank, runtime prefill scratch, or fixed-mask
         * publication without adding one-off logging at each caller.
         *
         * @return Stable key/value diagnostics intended for fatal logs and
         * PerfStats failure records.
         */
        std::string graphCaptureReadinessDebugString() const override;
        bool supportsGraphCaptureAfterLaunchPreparation() const override;
        bool supportsLazyPrefillGraphCapturePreflight() const override;
        bool supportsPaddedPrefillGraphCapturePreflight() const override;
        /**
         * @brief Preflight transfer-lane resources before a prefill graph launch.
         *
         * Transfer-backed LLEP is a multi-stream graph transaction. Preparing
         * the shared rolling lane here prevents stream/event allocation from
         * occurring inside stage execution while CUDA/HIP capture is active.
         */
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
        {
            return supportsGraphCaptureAfterLaunchPreparation() ||
                           requestsTransferBackedCurrentBatchPrefillLLEP() ||
                           params_.prefix_runtime_device_rehydration
                       ? GraphLaunchPreparationPolicy::CaptureOnly
                       : GraphLaunchPreparationPolicy::None;
        }
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
            runtime_grouped_decode_launch_state_prepared_ = false;
        }

        /**
         * @brief Invalidate MoE descriptor-table handles owned by kernel dynamic state.
         *
         * Grouped routed-expert prefill and decode stages cache integer table
         * IDs returned by the CUDA/HIP MoE backend. Those IDs are handles into
         * backend-owned dynamic descriptor tables, not model-weight ownership
         * and not request KV/GDN/MTP state. A hard kernel-dynamic reset clears
         * the backend tables while cached ComputeGraphs may retain this stage,
         * so the stage must forget the table IDs and require an eager rebuild
         * before any later capture or replay.
         */
        void invalidateKernelDynamicState() override
        {
            IMoEKernel *kernel =
                params_.routed_pipeline_kernel_owner &&
                        params_.routed_pipeline_kernel_owner->kernel
                    ? params_.routed_pipeline_kernel_owner->kernel.get()
                    : owned_moe_kernel_.get();
            if (kernel)
            {
                kernel->resetDynamicState();
                kernel->clearGPUStreamBinding();
            }
            grouped_gateup_desc_table_id_ = -1;
            grouped_gateup_desc_table_num_experts_ = 0;
            grouped_gateup_desc_table_d_model_ = 0;
            grouped_gateup_desc_table_intermediate_ = 0;
            grouped_gateup_desc_table_dirty_ = false;
            grouped_down_desc_table_id_ = -1;
            grouped_down_desc_table_num_experts_ = 0;
            grouped_down_desc_table_d_model_ = 0;
            grouped_down_desc_table_intermediate_ = 0;
            grouped_down_desc_table_dirty_ = false;
            combined_shared_gateup_desc_table_id_ = -1;
            combined_shared_gateup_desc_table_d_model_ = 0;
            combined_shared_gateup_desc_table_intermediate_ = 0;
            combined_shared_down_desc_table_id_ = -1;
            combined_shared_down_desc_table_d_model_ = 0;
            combined_shared_down_desc_table_intermediate_ = 0;
            combined_shared_desc_table_d_model_ = 0;
            combined_shared_desc_table_intermediate_ = 0;
            runtime_grouped_decode_launch_state_prepared_ = false;
            invalidateFixedTopologyMaskPublication();
        }

        /**
         * @brief Clear stream ownership without invalidating captured decode tables.
         *
         * The runtime grouped decode path warms descriptor and pointer-table
         * slots before graph capture. A preserved CUDA/HIP graph executable
         * still reads those device slots by address, so request-boundary replay
         * preservation must not flip runtime_grouped_decode_launch_state_prepared_ back to
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

        /**
         * @brief Return whether this stage owns grouped-verifier route scratch.
         *
         * GPU grouped verification must defer persistent routing-history
         * updates until the accepted-state transaction has produced its
         * device-resident row counts. The orchestrator uses this predicate to
         * discover exactly the verifier MoE stages that must participate in
         * that commit.
         */
        [[nodiscard]] bool
        requiresCommittedGroupedVerifierHistogramPublication() const noexcept
        {
            return params_.device_id.is_gpu() &&
                   params_.defer_grouped_verifier_histogram_publication;
        }

        /**
         * @brief Publish accepted grouped-verifier demand on its producer stream.
         *
         * This method is called once per layer from the accepted-state
         * publication transaction. It consumes the route ids and final
         * participant assignments retained by the verifier graph, combines
         * them with device-owned accepted prefix counts, and enqueues one
         * graph-capturable backend commit kernel. No host route or acceptance
         * data is read.
         *
         * @param accepted_state_counts_device Per-request committed row counts.
         * @param publication_ok_flags_device Per-request metadata validity.
         * @param request_count Active requests in the verifier graph.
         * @param rows_per_request Padded physical rows for each request.
         * @param producer_stream Exact accepted-publication CUDA/HIP stream.
         * @return true after the backend commit has been enqueued.
         */
        bool publishCommittedGroupedVerifierHistograms(
            const int32_t *accepted_state_counts_device,
            const int32_t *publication_ok_flags_device,
            int request_count,
            int rows_per_request,
            void *producer_stream);

        /**
         * @brief Enqueue the committed grouped-verifier histogram kernel only.
         *
         * Captured publication graphs must not mutate a host diagnostic stream
         * alias while their graph body is being warmed or captured: once the
         * native graph is cloned into a parent loop, that alias would still name
         * the old capture stream rather than the parent execution stream.  This
         * entry point performs the production mutation and nothing else.  A
         * terminal diagnostic owner may publish observation provenance after
         * the enclosing graph launch through a separate lifecycle API.
         */
        bool enqueueCommittedGroupedVerifierHistograms(
            const int32_t *accepted_state_counts_device,
            const int32_t *publication_ok_flags_device,
            int request_count,
            int rows_per_request,
            void *producer_stream);

        // Test accessor
        void setMoEKernelForTesting(IMoEKernel *kernel)
        {
            params_.routed_pipeline_kernel_owner.reset();
            owned_moe_kernel_.reset();
            moe_kernel_ = kernel;
        }
        void setRuntimeGroupedDecodeLaunchStatePreparedForTesting(bool warmed) { runtime_grouped_decode_launch_state_prepared_ = warmed; }
        /**
         * @brief Mark runtime-prefill scratch as prepared without performing GPU work.
         *
         * Unit tests use a host-backed runtime-table sentinel to exercise the
         * capture predicate.  Real execution may set this state only through
         * initializeMoERuntimeTableForGroupedPrefill(), which validates every
         * persistent device pointer and capacity before returning true.
         */
        void setRuntimePrefillGroupingAvailableForTesting(bool available)
        {
            moe_prefill_runtime_grouping_available_ = available;
        }
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
        bool usesFixedTopologyGroupedPrefillForTesting() const
        {
            return canUseFixedTopologyGroupedPrefill();
        }
        bool hasPublishedFixedTopologyMaskForTesting() const noexcept
        {
            return fixed_topology_mask_publication_state_ ==
                   FixedTopologyMaskPublicationState::Published;
        }
        std::vector<int> fixedTopologyPrefillExpertIdsForTesting() const
        {
            return fixedTopologyPrefillExpertIds();
        }
        std::vector<uint8_t> fixedTopologyPrefillExpertMaskBytesForTesting() const
        {
            return fixedTopologyPrefillExpertMaskBytes();
        }
        TensorBase *combinedSharedGateInputForTesting() const
        {
            return effectiveSafeCompositeSharedGateInput();
        }
        void bindPreparedExpertEnginesForTesting(const std::vector<int> &expert_ids)
        {
            bindPreparedExpertEnginesForExperts(expert_ids);
        }
        void addPendingGpuDirectTransferForTesting(GpuDirectTransferCompletion completion)
        {
            addPendingGpuDirectTransfer(std::move(completion));
        }
        size_t pendingGpuDirectTransferCountForTesting() const
        {
            return pending_gpu_direct_transfers_.size();
        }
        void addPendingGpuDirectTransfersFromStoreForTesting(const std::vector<int> &expert_ids)
        {
            addPendingGpuDirectTransfersFromStore(expert_ids);
        }

    private:
        /**
         * @brief Lifecycle state for the graph-stable fixed-topology expert mask.
         *
         * A mask is ordinary control-plane metadata until it has been copied to
         * the backend-owned workspace buffer. Graph execution may consume only
         * the Published state. Workspace rebinding, kernel dynamic-state reset,
         * and placement changes move the stage back to NeedsPublication so stale
         * device addresses or stale ownership can never look capture-ready.
         */
        enum class FixedTopologyMaskPublicationState : uint8_t
        {
            NotRequired,
            NeedsPublication,
            Published,
        };

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

        /**
         * @brief Independently owned launch state for this routed-expert stage.
         *
         * A MoE backend caches streams, workspaces, descriptor tables, and
         * scratch pointers. Keeping it stage-local prevents a concurrently
         * captured MTP, routing, shared-expert, or maintenance stage from
         * changing those values between grouped launches.
         */
        mutable std::unique_ptr<IMoEKernel> owned_moe_kernel_;
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
        mutable bool runtime_grouped_decode_launch_state_prepared_ = false;

        /// Explicit publication state for the backend-owned fixed-topology mask.
        FixedTopologyMaskPublicationState fixed_topology_mask_publication_state_ =
            FixedTopologyMaskPublicationState::NotRequired;

        /**
         * @brief Resolve the sole output owned by this routed-expert producer.
         *
         * Ordinary routed execution publishes the final MoE output directly.
         * LocalTP canonical execution instead publishes one tensor row per
         * original router slot; a later reducer owns the final output. Keeping
         * that choice behind one method prevents individual M=1, grouped-M,
         * runtime-table, and fixed-table exits from publishing the inactive
         * reducer destination.
         *
         * @return Non-null tensor written by the current expert transaction.
         * @throws std::logic_error if graph construction omitted both output
         *         contracts.
         */
        ITensor *publicationTarget() const;

        /**
         * @brief Publish the routed-expert producer's exact output event.
         *
         * The backend kernel and this stage share the same explicit producer
         * stream. This method records the stage-level handoff only for the
         * tensor returned by publicationTarget(); it never allocates storage,
         * substitutes a stream, or touches the later canonical reducer target.
         */
        void publishPublicationTarget() const;

        /// Fast path for decode (seq_len=1): avoids token grouping, gather/scatter,
        /// and per-expert heap allocations. Uses routing results directly.
        bool executeSingleToken(IDeviceContext *ctx);

        /**
         * @brief Execute verifier-sized batches with grouped serial-row math.
         *
         * MTP state publication restores live KV/GDN/conv state from a selected
         * verifier row.  That remains sound only when each grouped row is
         * numerically equivalent to ordinary one-token decode.  M=1 therefore
         * enters the normal decode route, while M=2..4 must use grouped
         * verifier implementations that preserve per-row projection math and
         * top-k accumulation order.  This function refuses unsupported GPU
         * requests rather than hiding them behind row replay.
         */
        bool executeDecodeEquivalentVerifierPrefill(IDeviceContext *ctx);

        /**
         * @brief CPU grouped verifier executor for routed MoE experts.
         *
         * The CPU route keeps sparse MoE economics by batching route slots by
         * expert, using the CPU NativeVNNI/floating grouped verifier GEMV hooks
         * for gate/up and SwiGLU/down work.  It stores every route-slot result
         * separately and performs the final row accumulation in original top-k
         * order so the output is serial-decode equivalent without calling the
         * single-row stage in a loop.
         */
        bool executeCPUGroupedDecodeEquivalentVerifierPrefill(IDeviceContext *ctx);

        void ensureGemmEnginesCached();
        bool ensureGemmEnginesForExperts(const std::vector<int> &expert_ids);
        void bindPreparedExpertEnginesForExperts(const std::vector<int> &expert_ids);
        void addPendingGpuDirectTransfer(GpuDirectTransferCompletion completion);
        void addPendingGpuDirectTransfersFromStore(const std::vector<int> &expert_ids);
        bool waitForPendingGpuDirectTransfers();
        bool refreshGraphStablePlacement(bool preserve_capture_ready);
        /**
         * @brief Revalidates the graph-owned one-token MoE runtime placement.
         *
         * Dynamic routed-expert and overlay decode stages may receive a
         * graph-initialized runtime table from Qwen35MoEGraph instead of
         * synthesizing a full-owner table inside the stage.  This refresh keeps
         * descriptor-table readiness, the active placement bank, and the
         * stage-local initialized flag in lockstep so a later warmup/replay does
         * not re-enter the forbidden synthesis path for explicit owner metadata.
         *
         * @param preserve_capture_ready Preserve an already-warmed capture-ready
         * state only when all refreshed resources still match.
         * @return true when the active runtime bank is usable or this stage does
         * not participate in graph-stable runtime decode placement.
         */
        bool refreshRuntimeGroupedDecodePlacement(bool preserve_capture_ready);
        bool refreshFixedTopologyGroupedPrefillPlacement();
        /**
         * @brief Mark the current fixed-topology mask as requiring publication.
         *
         * This transition is called whenever the workspace, kernel dynamic
         * state, expert ownership, or replica ownership changes. Full-ownership
         * stages do not need a mask and remain in NotRequired.
         */
        void invalidateFixedTopologyMaskPublication() noexcept;
        /**
         * @brief Publish the host control-plane mask to backend-owned device storage.
         *
         * Publication is legal only outside graph capture and is ordered on the
         * stage's explicit stream. The subsequent warmup grouping launch observes
         * that copy on the same stream; graph capture is admitted only after this
         * method has transitioned the stage to Published.
         */
        bool publishFixedTopologyMaskBeforeCapture(IMoEKernel *kernel);
        bool ensureGroupedGateUpDescriptorTable(IMoEKernel *kernel, int d_model, int intermediate);
        bool ensureGroupedDownDescriptorTable(IMoEKernel *kernel, int d_model, int intermediate);
        bool ensureCombinedSharedVerifierResources(IMoEKernel *kernel, int d_model, int intermediate);
        bool initializeMoERuntimeTableForGroupedDecode();
        bool initializeMoERuntimeTableForGroupedPrefill();
        bool initializeFixedTopologyGroupedPrefill();
        int expectedGroupedDecodeParticipantCount() const;
        bool runtimeTableHasActiveGroupedDecodeBank() const;
        bool supportsRequestedRoutedAssignmentPolicy() const;
        bool canUseRuntimePrefillGrouping() const;
        bool canUseFixedTopologyGroupedPrefill() const;
        /**
         * @brief True when verifier rows can use the safe routed+shared composite path.
         *
         * The rejected shortcut treated the shared expert as an extra routed expert
         * in one MoE prefill table.  This predicate guards the replacement design:
         * routed experts use the proven grouped verifier pipeline, shared expert
         * rows use the same grouped table-prefill route as standalone shared
         * verifier rows, and the stage combines those two branch outputs with
         * the normal shared-gate add kernel.
         */
        bool canUseSafeCombinedSharedVerifierComposite() const;
        TensorBase *effectiveSafeCompositeSharedGateInput() const;
        bool executeSafeCombinedSharedVerifierComposite(IMoEKernel *kernel) const;
        bool executeFixedTopologyGroupedPrefill(IMoEKernel *kernel, int max_tokens);
        bool requestsTransferBackedCurrentBatchPrefillLLEP() const noexcept;
        bool hasValidCompactLLEPTransferBinding() const;
        bool hasTransferBackedPrefillLLEP() const;
        bool executeTransferBackedPrefillLLEPMovement(
            IMoEKernel *kernel,
            DeviceMoERebalanceStatus **transfer_status_out,
            DeviceMoERebalanceApplyStatus **apply_status_out,
            PrefillLLEPTransferPurpose purpose,
            DeviceMoERebalanceTransferState *transfer_state_override =
                nullptr) const;
        bool isDeviceRoutedDecodeGraphCapturable() const;
        bool supportsFixedTopologyPrefillGraphCapturePreflight() const;
        bool isFixedTopologyPrefillGraphCapturable() const;
        const std::vector<bool> *fixedTopologyPrefillMask() const;
        bool hasFixedTopologyPrefillExpertMask() const;
        bool usesMaskedFixedTopologyPrefill() const;
        /**
         * @brief True when grouped prefill consumes the published fixed mask.
         *
         * Runtime-table grouping and fixed-mask grouping are mutually exclusive
         * placement sources.  LLEP and other runtime-grouped paths consume the
         * persistent DeviceMoERuntimeTable directly, so requiring or publishing
         * the static mask for those paths creates a contradictory lifecycle
         * contract.  Centralizing the distinction here keeps execution,
         * invalidation, and graph-capture readiness in agreement.
         */
        bool usesPublishedFixedTopologyMaskGrouping() const;
        std::vector<int> fixedTopologyPrefillExpertIds() const;
        std::vector<uint8_t> fixedTopologyPrefillExpertMaskBytes() const;
        bool expertComputesLocally(int expert_id) const;
        bool hasFullLocalExpertOwnership() const;
        bool expertMaskAllEnabled() const;
        bool hasAllPreparedExpertGemmEngines() const;
        /**
         * @brief Verify prepared GEMM ownership without requiring remote slots.
         *
         * Prepared expert arrays are indexed by global expert id so descriptor
         * tables can preserve the model's routing namespace on every
         * participant. Static apportioned and runtime-balanced topologies leave
         * non-local entries null by design. This predicate therefore validates
         * all three projections only for experts selected by
         * expertComputesLocally(), while still requiring the arrays themselves
         * to span the complete global expert namespace.
         *
         * @return True when every locally computable expert has gate, up, and
         * down GEMM engines ready for descriptor publication.
         */
        bool hasPreparedExpertGemmEnginesForLocalOwnership() const;
        bool hasPreparedExpertGemmEnginesForExperts(const std::vector<int> &expert_ids) const;
        bool hasGroupedDecodeDescriptorExportSupport() const;
        const DeviceMoEPlacementBank *activeRuntimePlacementBank() const;
        bool runtimeLocalComputeEnabled(const DeviceMoEPlacementBank *bank, int expert_id) const;
        void ensureScratchBuffers(int max_batch) const;
        IMoEKernel *ensureMoEKernel() const;

        mutable int grouped_gateup_desc_table_id_ = -1;
        mutable int grouped_gateup_desc_table_num_experts_ = 0;
        mutable int grouped_gateup_desc_table_d_model_ = 0;
        mutable int grouped_gateup_desc_table_intermediate_ = 0;
        bool grouped_gateup_desc_table_dirty_ = false;

        mutable int grouped_down_desc_table_id_ = -1;
        mutable int grouped_down_desc_table_num_experts_ = 0;
        mutable int grouped_down_desc_table_d_model_ = 0;
        mutable int grouped_down_desc_table_intermediate_ = 0;
        bool grouped_down_desc_table_dirty_ = false;

        mutable int combined_shared_desc_table_d_model_ = 0;
        mutable int combined_shared_desc_table_intermediate_ = 0;
        mutable int combined_shared_gateup_desc_table_id_ = -1;
        mutable int combined_shared_gateup_desc_table_d_model_ = 0;
        mutable int combined_shared_gateup_desc_table_intermediate_ = 0;
        mutable int combined_shared_down_desc_table_id_ = -1;
        mutable int combined_shared_down_desc_table_d_model_ = 0;
        mutable int combined_shared_down_desc_table_intermediate_ = 0;
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
        std::vector<GpuDirectTransferCompletion> pending_gpu_direct_transfers_;
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
            /**
             * @brief Arena-owned gate projection scratch.
             *
             * GPU shared-expert execution must never allocate this tensor from
             * execute(). The graph resolver sizes one reusable MoE scratch pair
             * for the largest routed/shared intermediate width, and the stage
             * contract makes the executor establish its device storage before
             * any projection launch.
             */
            TensorBase *gate_scratch = nullptr; ///< [capacity_rows, capacity_intermediate]
            /**
             * @brief Arena-owned up projection scratch.
             *
             * This buffer has the same graph lifetime and capacity contract as
             * @ref gate_scratch. Separate buffers are required because fused
             * gate/up projection streams may write them concurrently.
             */
            TensorBase *up_scratch = nullptr; ///< [capacity_rows, capacity_intermediate]
            int seq_len = 0;
            int d_model = 0;
            int intermediate = 0;

            BufferId input_buffer_id = BufferId::NORMALIZED;
            BufferId output_buffer_id = BufferId::MOE_SHARED_EXPERT_OUTPUT;
            BufferId gate_scratch_buffer_id = BufferId::MOE_GATE_SCRATCH;
            BufferId up_scratch_buffer_id = BufferId::MOE_UP_SCRATCH;
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
        bool supportsGraphCaptureAfterLaunchPreparation() const override;
        std::string graphCaptureReadinessDebugString() const override;
        bool supportsLazyPrefillGraphCapturePreflight() const override;
        bool supportsPaddedPrefillGraphCapturePreflight() const override;
        /**
         * @brief Bind shared-expert kernels and immutable grouped-decode metadata.
         *
         * This preparation replaces the historical eager arithmetic pass. It
         * resolves persistent GEMM/MoE engines, validates arena-owned scratch,
         * publishes fixed descriptor/pointer tables, and executes no model math.
         */
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
        {
            return params_.device_id.is_gpu() &&
                           supportsGraphCaptureAfterLaunchPreparation()
                       ? GraphLaunchPreparationPolicy::CaptureOnly
                       : GraphLaunchPreparationPolicy::None;
        }
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
            grouped_decode_launch_state_prepared_ = false;
        }

        /**
         * @brief Invalidate shared-expert descriptor-table handles after kernel reset.
         *
         * The shared-expert grouped decode route uses the same backend table
         * lifetime as routed MoE: descriptor table IDs belong to kernel-dynamic
         * state, while this stage object and its prepared GEMM engines can
         * survive graph-cache reuse. Forgetting the handles here forces the
         * next eager execution to upload fresh tables before capture.
         */
        void invalidateKernelDynamicState() override
        {
            if (owned_moe_kernel_)
            {
                owned_moe_kernel_->resetDynamicState();
                owned_moe_kernel_->clearGPUStreamBinding();
            }
            shared_grouped_gateup_desc_table_id_ = -1;
            shared_grouped_gateup_desc_table_d_model_ = 0;
            shared_grouped_gateup_desc_table_intermediate_ = 0;
            shared_grouped_down_desc_table_id_ = -1;
            shared_grouped_down_desc_table_d_model_ = 0;
            shared_grouped_down_desc_table_intermediate_ = 0;
            grouped_decode_launch_state_prepared_ = false;
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
        bool usesDecodeEquivalentVerifierPrefillForTesting() const;
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

        /**
         * @brief Typed aliases of the arena-owned projection scratch tensors.
         *
         * The stage deliberately does not own these pointers. BufferArena owns
         * their host/device allocations for the complete graph lifetime, which
         * lets every layer reuse the same stable addresses without one
         * runtime device allocation per layer during prefill.
         */
        FP32Tensor *scratch_gate_ = nullptr;
        FP32Tensor *scratch_up_ = nullptr;
        /**
         * @brief CPU-only construction-time storage for standalone stages.
         *
         * Production GPU graphs must supply arena-owned scratch through Params.
         * CPU unit and standalone execution can own ordinary host tensors
         * because no device allocation or graph-address lifetime is involved.
         */
        std::shared_ptr<FP32Tensor> owned_cpu_scratch_gate_;
        std::shared_ptr<FP32Tensor> owned_cpu_scratch_up_;
        mutable int scratch_seq_len_ = 0;
        /**
         * @brief True once explicit preparation has populated grouped-decode state.
         *
         * The CUDA grouped decode kernels read device-side arrays of scratch
         * tensor pointers during graph replay. prepareGraphLaunch() publishes
         * those arrays without running expert arithmetic; capture is admitted
         * only for the exact current scratch/workspace owner.
         */
        mutable bool grouped_decode_launch_state_prepared_ = false;

        void ensureGemmEnginesCached() const;
        bool ensureSharedGroupedGateUpDescriptorTable(IMoEKernel *kernel, int d_model, int intermediate) const;
        bool ensureSharedGroupedDownDescriptorTable(IMoEKernel *kernel, int d_model, int intermediate) const;
        bool shouldUseGroupedVerifierPrefillRoute() const;
        bool shouldUseDecodeEquivalentVerifierPrefill() const;
        bool shouldUseGroupedDecodeRoute() const;
        bool tryGroupedVerifierPrefill(IMoEKernel *kernel, int d_model, int intermediate) const;
        bool tryGroupedDecode(IMoEKernel *kernel, int d_model, int intermediate) const;
        /**
         * @brief Validate the graph-owned scratch capacity before execution.
         *
         * GPU callers fail closed when the graph omitted, mistyped, or
         * undersized either tensor. This method never allocates or resizes
         * storage, so a malformed graph cannot trigger a hot-path allocation.
         */
        bool validatePlannedScratch(int rows, int intermediate) const;

        /**
         * @brief Execute shared expert verifier rows with grouped decode math.
         *
         * The shared expert has no routing state, but its quantized GEMM kernels
         * can still choose verifier-row math that differs from ordinary decode.
         * M=1 uses the normal decode path.  M=2..4 use backend grouped verifier
         * projection and SwiGLU/down hooks, or the GPU grouped table-prefill
         * route, so production never loops over row replay.
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

        /**
         * @brief Shared-FFN-stage-owned MoE launch and descriptor state.
         */
        mutable std::unique_ptr<IMoEKernel> owned_moe_kernel_;
        mutable IMoEKernel *moe_kernel_ = nullptr;
        IMoEKernel *ensureMoEKernel() const;

    public:
        // Test accessors
        void setMoEKernelForTesting(IMoEKernel *kernel)
        {
            owned_moe_kernel_.reset();
            moe_kernel_ = kernel;
        }
        void setScratchSeqLenForTesting(int n) { scratch_seq_len_ = n; }
        void setGroupedDecodeLaunchStatePreparedForTesting(bool warmed) { grouped_decode_launch_state_prepared_ = warmed; }
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
            /**
             * @brief Device-owned logical row count for variable-row GPU graphs.
             *
             * Request admission and grouped-verifier preparation publish this
             * stable scalar before graph consumption. The shared-gate kernel
             * masks the physical suffix directly from device memory on every
             * capture and replay; the stage never mirrors or uploads its value.
             */
            const int32_t *active_row_count_device = nullptr;

            BufferId input_buffer_id = BufferId::NORMALIZED;
            BufferId output_buffer_id = BufferId::MOE_SHARED_EXPERT_OUTPUT;
            BufferId residual_buffer_id = BufferId::MOE_COMBINED_OUTPUT;
            BufferId combined_output_buffer_id = BufferId::ATTN_PROJ;
        };

        /**
         * @brief Construct the shared-expert sigmoid gate stage.
         * @param params Immutable weights, graph buffers, and device row geometry.
         * @throws std::invalid_argument if a supplied input gate is not FP32;
         *         model-weight preparation owns that normalization on every backend.
         */
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
        bool supportsGraphCaptureAfterLaunchPreparation() const override;
        bool supportsLazyPrefillGraphCapturePreflight() const override;
        bool supportsPaddedPrefillGraphCapturePreflight() const override;
        bool supportsPaddedPrefillRealLengthContract() const override;
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
        {
            return params_.device_id.is_gpu() &&
                           supportsGraphCaptureAfterLaunchPreparation()
                       ? GraphLaunchPreparationPolicy::CaptureOnly
                       : GraphLaunchPreparationPolicy::None;
        }
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;
        void unbindWorkspace() override;
        bool hasWorkspace() const override;
        DeviceWorkspaceManager *getWorkspace() const override;

        /**
         * @brief Reset the private backend object after a hard graph reset.
         */
        void invalidateKernelDynamicState() override;

    private:
        Params params_;

        TensorBase *effectiveGateInput() const;
        bool gateInputReadyForGraphCapture() const;

        /**
         * @brief Shared-gate-stage-owned MoE launch state.
         */
        mutable std::unique_ptr<IMoEKernel> owned_moe_kernel_;
        mutable IMoEKernel *moe_kernel_ = nullptr;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;
        IMoEKernel *ensureMoEKernel() const;

    public:
        // Test accessors
        void setMoEKernelForTesting(IMoEKernel *kernel)
        {
            owned_moe_kernel_.reset();
            moe_kernel_ = kernel;
        }

        /** @brief Expose the captured device row-count owner to graph tests. */
        const int32_t *activeRowCountDeviceForTesting() const noexcept
        {
            return params_.active_row_count_device;
        }
    };

    /**
     * @brief Device-only canonical LocalTP routed-expert reduction.
     *
     * Expert placement is intentionally absent from this stage. Each input
     * slot has already been reduced to one fixed collective root, so only that
     * participant walks router slots in increasing order and overwrites the
     * compact routed output. Non-root participants execute a declarative no-op
     * at the same graph position and receive the result from the following
     * broadcast. Static, Dynamic, LLEP, and prefix-restored placement therefore
     * share one visible FP32 addition tree without redundant per-device work or
     * host orchestration.
     */
    class MoECanonicalRouteReduceStage final : public IComputeStage
    {
    public:
        /** @brief Immutable graph-bound reducer parameters. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            TensorBase *canonical_route_contributions = nullptr;
            TensorBase *output = nullptr;
            int seq_len = 0;
            int top_k = 0;
            int d_model = 0;
            /** Communicator-local participant represented by this graph. */
            int participant_device_index = -1;
            /** Fixed participant that owns the ordered reduction kernel. */
            int root_device_index = -1;
            BufferId canonical_route_contributions_buffer_id =
                BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS;
            BufferId output_buffer_id = BufferId::MOE_COMBINED_OUTPUT;
        };

        explicit MoECanonicalRouteReduceStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override
        {
            return ComputeStageType::MOE_CANONICAL_ROUTE_REDUCE;
        }
        std::string name() const override
        {
            return "moe_canonical_route_reduce";
        }
        size_t estimatedFlops() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override;
        bool supportsGraphCaptureAfterLaunchPreparation() const override;
        /**
         * @brief Admit cold exact-shape prefill before the reducer owns its kernel wrapper.
         *
         * The reducer's launch geometry depends only on the graph-bound tensor
         * dimensions. Its first eager warmup constructs the backend wrapper and
         * launches the same fixed-grid device kernel that capture records on the
         * following request. No descriptor table, transfer, or host result is
         * part of this initialization boundary.
         */
        bool supportsLazyPrefillGraphCapturePreflight() const override;
        /**
         * @brief Admit padded buckets because reduction has no persistent row state.
         *
         * Padding contributes only to padding output rows. The reducer neither
         * advances KV/recurrent state nor reads a host-visible effective length,
         * so the fixed bucket geometry is graph-stable and semantically isolated
         * from the real prompt prefix.
         */
        bool supportsPaddedPrefillGraphCapturePreflight() const override;
        /**
         * @brief Resolve the root reducer kernel before native graph capture.
         *
         * Non-root participants own no arithmetic. The root binds only the
         * persistent backend wrapper and exact stream; no tensor value is read or
         * written by preparation.
         */
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
        {
            return supportsGraphCaptureAfterLaunchPreparation()
                       ? GraphLaunchPreparationPolicy::CaptureOnly
                       : GraphLaunchPreparationPolicy::None;
        }
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;
        /**
         * @brief Declare coherence only on the participant that runs arithmetic.
         *
         * The non-root node exists solely to keep the LocalTP DAG symmetric. It
         * owns no buffers and performs no memory access, so asking the executor
         * to apply FULL coherence there would contradict its empty declarative
         * contract. The root retains ordinary input/output coherence around the
         * production reduction kernel.
         */
        CoherencePolicy coherencePolicy() const override
        {
            return params_.participant_device_index ==
                           params_.root_device_index
                       ? CoherencePolicy::FULL
                       : CoherencePolicy::NONE;
        }
        StageDumpInfo buildDumpInfoImpl() const override;

        /**
         * @brief Return immutable graph-bound reducer parameters for diagnostics.
         * @return Canonical input/output identities and fixed reduction geometry.
         */
        [[nodiscard]] const Params &params() const { return params_; }

    private:
        Params params_;
        mutable std::unique_ptr<IMoEKernel> owned_moe_kernel_;
        mutable IMoEKernel *moe_kernel_ = nullptr;
    };

} // namespace llaminar2
