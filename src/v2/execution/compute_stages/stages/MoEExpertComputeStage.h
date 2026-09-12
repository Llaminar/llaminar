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
#include "MoEOverlayPinnedRouteEvidenceViews.h"
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
#include "../../moe/IMoEGroupedVerifierHistogramPublisher.h"

#include <memory>
#include <atomic>
#include <cstdint>
#include <span>
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
    class ITPContext;
    class ILocalTPContext;
    class DeviceMoERebalanceTransferState;
    class MoEOverlayNodeLocalRouteExchange;
    class MoEOverlayPersistentGraphStorage;

    /**
     * @brief Setup-owned CPU grouped-MoE scratch shared by one serial graph family.
     *
     * Heterogeneous ExpertOverlay executes model layers, prefill buckets, and
     * MTP verifier roles in strict transaction order for one CPU participant.
     * Those mutually exclusive stages borrow this one first-touched workspace
     * instead of allocating and retaining a worst-case gate/up/down arena per
     * layer. An RAII lease turns the serial-family proof into a runtime
     * invariant: concurrent use is a fatal ownership error, never an implicit
     * allocation or private fallback.
     */
    class CPUGroupedMoESerialWorkspace final
    {
    public:
        /** @brief Complete immutable capacity contract. */
        struct Config
        {
            size_t row_capacity = 0; ///< Maximum compact input rows.
            int d_model = 0; ///< Hidden/output width.
            int expert_intermediate = 0; ///< Routed-expert SwiGLU width.
            int num_experts = 0; ///< Logical expert-table width.
            int routing_top_k = 0; ///< Maximum routes admitted per row.
            std::string debug_name; ///< Stable diagnostics identity.
        };

        /** @brief Exclusive invocation lease for one serial participant stage. */
        class InvocationLease final
        {
        public:
            InvocationLease() = default;
            ~InvocationLease();
            InvocationLease(const InvocationLease &) = delete;
            InvocationLease &operator=(const InvocationLease &) = delete;
            InvocationLease(InvocationLease &&other) noexcept;
            InvocationLease &operator=(InvocationLease &&other) noexcept;

        private:
            friend class CPUGroupedMoESerialWorkspace;
            explicit InvocationLease(
                CPUGroupedMoESerialWorkspace *owner) noexcept
                : owner_(owner)
            {
            }
            void release() noexcept;
            CPUGroupedMoESerialWorkspace *owner_ = nullptr;
        };

        /**
         * @brief Allocate and first-touch the complete serial workspace.
         * @throws std::invalid_argument for incomplete or overflowing geometry.
         */
        explicit CPUGroupedMoESerialWorkspace(Config config);
        ~CPUGroupedMoESerialWorkspace();

        /**
         * @brief Price the exact payload allocated by this workspace before construction.
         * @param config Same immutable geometry consumed by the constructor.
         * @return FP32, Q8 and kernel-publication bytes, without allocating anything.
         * @throws std::invalid_argument for incomplete geometry.
         * @throws std::overflow_error for unrepresentable row or byte counts.
         */
        [[nodiscard]] static size_t plannedAllocationBytes(const Config &config);

        CPUGroupedMoESerialWorkspace(
            const CPUGroupedMoESerialWorkspace &) = delete;
        CPUGroupedMoESerialWorkspace &operator=(
            const CPUGroupedMoESerialWorkspace &) = delete;

        /** @brief Check whether one runtime invocation fits this setup plan. */
        [[nodiscard]] bool supports(
            int rows,
            int route_width,
            int d_model,
            int expert_intermediate,
            int num_experts) const noexcept;

        /**
         * @brief Acquire exclusive access for one participant-local packet.
         * @throws std::invalid_argument when runtime geometry exceeds the plan.
         * @throws std::logic_error when two allegedly serial stages overlap.
         */
        [[nodiscard]] InvocationLease acquire(
            int rows,
            int route_width,
            int d_model,
            int expert_intermediate,
            int num_experts,
            int layer_idx);

        /** @brief Return all setup-owned payload bytes. */
        [[nodiscard]] size_t allocationBytes() const noexcept
        {
            return allocation_bytes_;
        }

    private:
        friend class MoEExpertComputeStage;

        size_t row_capacity_ = 0;
        size_t route_capacity_ = 0;
        int d_model_ = 0;
        int expert_intermediate_ = 0;
        int num_experts_ = 0;
        int routing_top_k_ = 0;
        std::string debug_name_;
        size_t allocation_bytes_ = 0;
        std::atomic_flag invocation_active_ = ATOMIC_FLAG_INIT;

        std::shared_ptr<FP32Tensor> scratch_batch_;
        std::shared_ptr<FP32Tensor> scratch_gate_;
        std::shared_ptr<FP32Tensor> scratch_up_;
        std::shared_ptr<FP32Tensor> scratch_out_;
        std::vector<Q8_1Block> router_q8_;
        std::vector<Q8_1Block> swiglu_q8_;
        std::vector<float> route_outputs_;
        std::unique_ptr<IMoEKernel> moe_kernel_;
    };

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
     * @ref GraphPhasedCurrentBatch after its routed-row economy gate has
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
         * Consume assignments published by explicit plan/sideband/apply graph
         * stages before this expert-compute node executes.
         *
         * Graph construction may select this only for an amortizable long
         * prefill in a homogeneous graph-capturable LocalTP domain. The expert
         * stage never launches the payload collective in this mode.
         */
        GraphPhasedCurrentBatch = 1,
    };

    /**
     * @brief Select the authority that publishes CPU NativeVNNI input rows.
     *
     * Ordinary CPU graphs require the immediately preceding CPU router to own
     * the canonical Q8_1 publication. A heterogeneous ExpertOverlay participant
     * instead receives authoritative FP32 rows through its fixed ticket and
     * explicitly publishes the same canonical CPU representation at that
     * declared transport boundary.
     */
    enum class CPURouterQ8InputPublicationPolicy : uint8_t
    {
        RequireRouterStage = 0,
        PublishTransportedRows = 1,
    };

    /**
     * @brief Select the sole graph-build authority for routed-expert weights.
     *
     * Ordinary standalone graphs may prepare engines from their bound raw
     * three-dimensional parent tensors. ExpertOverlay graphs instead receive
     * participant-scoped engines from @ref ExpertGemmRegistry; consulting a
     * generic raw-parent binding there can select another participant's packed
     * slice when CPU and GPU tiers share a model layer.
     */
    enum class MoEExpertWeightResolutionPolicy : uint8_t
    {
        RawOrPrepared = 0,
        PreparedRegistryOnly = 1,
    };

    /**
     * @brief Select when a registry-only stage receives its first live engine bank.
     *
     * Ordinary graph stages are born with a complete immutable bank. A serial
     * CPU ExpertOverlay endpoint is different: graph construction creates one
     * retained executor before the participant residency authority publishes
     * its initial epoch. That executor is setup-primed with an explicitly empty
     * mask and cannot execute until a non-zero host-packet publication binds a
     * complete triplet for every active expert.
     */
    enum class MoEPreparedEngineBindingLifecycle : uint8_t
    {
        CompleteAtConstruction = 0,
        SparseOverlayInvocationPublished = 1,
    };

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
    class MoEExpertComputeStage : public IComputeStage,
                                  public IWorkspaceConsumer,
                                  public IMoEGroupedVerifierHistogramPublisher
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

            /**
             * @brief Authority for the CPU NativeVNNI hidden-row publication.
             *
             * Graph-native heterogeneous local-expert stages set
             * PublishTransportedRows after compact ticket dispatch. Both the
             * serial one-row and grouped-row NativeVNNI paths publish the
             * transported bytes before consuming them. Every ordinary CPU
             * graph retains RequireRouterStage and fails if its router/expert
             * publication edge is missing.
             */
            CPURouterQ8InputPublicationPolicy cpu_router_q8_input_publication =
                CPURouterQ8InputPublicationPolicy::RequireRouterStage;

            /**
             * @brief Serial participant workspace for CPU ExpertOverlay packets.
             *
             * Production heterogeneous CPU endpoints bind the workspace owned
             * by their compact-buffer arena. Ordinary CPU graphs leave this
             * empty and retain their graph-local scratch ownership.
             */
            std::shared_ptr<CPUGroupedMoESerialWorkspace>
                cpu_grouped_serial_workspace;

            // Expert weights (3D packed tensors) — used by CPU path
            TensorBase *gate_exps = nullptr; ///< [num_experts, intermediate, d_model]
            TensorBase *up_exps = nullptr;   ///< [num_experts, intermediate, d_model]
            TensorBase *down_exps = nullptr; ///< [num_experts, d_model, intermediate]
            int expert_intermediate = 0;

            /**
             * @brief Authority used to resolve executable expert matrices.
             *
             * PreparedRegistryOnly requires raw parent pointers to be absent
             * and a complete prepared triplet for every locally executable
             * expert. This makes an ExpertOverlay graph unable to fall back to
             * a coincidentally bound slice owned by another tier participant.
             */
            MoEExpertWeightResolutionPolicy expert_weight_resolution_policy =
                MoEExpertWeightResolutionPolicy::RawOrPrepared;

            /**
             * @brief Typed construction/publication boundary for prepared engines.
             *
             * SparseOverlayInvocationPublished is legal only for a serial CPU
             * transported-row endpoint with setup-owned grouped scratch. The
             * constructor accepts only an all-disabled mask in that state;
             * @ref bindSparseOverlayInvocation validates the first live bank.
             */
            MoEPreparedEngineBindingLifecycle prepared_engine_binding_lifecycle =
                MoEPreparedEngineBindingLifecycle::CompleteAtConstruction;

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

            // Per-expert 2D tensor views — used by raw-parent GPU paths
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
             * @brief Declare this expert stage's verifier-history lifecycle role.
             *
             * Sidecar and main verifier stages both use grouped kernels, but
             * only the selected main-target boundary may publish accepted rows
             * into decode maintenance history. Static placement names the same
             * boundary without retaining routes or creating a producer stream;
             * sidecars and non-owning overlay stages remain @c NotOwner.
             */
            MoEGroupedVerifierHistogramRole grouped_verifier_histogram_role =
                MoEGroupedVerifierHistogramRole::NotOwner;

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
             * The graph-selected publication layout determines whether this is
             * a dense original-slot tensor or an indexed packed-record buffer.
             * The accompanying arithmetic policy states whether rows are
             * already weighted or remain raw expert outputs. The stage never
             * collapses participant-local routes into @ref output when this
             * target is present; a following collective and canonical reducer
             * own the globally ordered fold.
             */
            TensorBase *canonical_route_contributions = nullptr;

            /** Arithmetic represented by each canonical route slot. */
            MoECanonicalRouteArithmeticPolicy canonical_route_arithmetic =
                MoECanonicalRouteArithmeticPolicy::Unspecified;

            /** Physical storage/wire layout of canonical route evidence. */
            MoECanonicalRoutePublicationLayout canonical_route_layout =
                MoECanonicalRoutePublicationLayout::Unspecified;

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
            /**
             * @brief Optional observation-only service telemetry capability.
             *
             * Ordinary runtime-table-backed stages derive this view from
             * @ref moe_runtime_table. Retained sparse endpoints bind it
             * explicitly because their immutable residency bank, not the
             * runtime table, is their sole placement authority.
             */
            DeviceMoEOverlayServiceTelemetryBinding
                overlay_service_telemetry;
            /**
             * @brief Immutable service phase owned by this graph family.
             *
             * Participant-local retained and inline sparse graphs set this at
             * construction because row geometry cannot distinguish ordinary
             * decode from an MTP predictor. Leave Auto only when ordinary
             * dense graph geometry is authoritative or when
             * @ref runtime_service_graph_role_device supplies a replay-varying
             * device-owned role.
             */
            MoEOverlayServicePhaseHint service_phase =
                MoEOverlayServicePhaseHint::Auto;
            /**
             * @brief Optional authenticated device-owned transaction role.
             *
             * Mapped heterogeneous follower graphs reuse one captured row
             * shape for decode, prefill, and grouped verification. Their
             * endpoint-private activation grant owns the current role, so
             * service telemetry reads this exact device address instead of
             * baking a phase into capture or accepting a host-side update.
             */
            const MoEOverlayInferenceGraphRole
                *runtime_service_graph_role_device = nullptr;
            /**
             * Route grouped rows through the persistent runtime-table grouper.
             *
             * This mechanism is shared by ordinary prefill and grouped MTP
             * verification.  Assignment policy remains an independent axis:
             * verifier rows use static ownership, while an explicitly selected
             * ordinary-prefill graph may use current-batch least-loaded
             * assignment.
             */
            bool use_runtime_row_grouping = false;

            /// LocalTP context for prefix-runtime payload rehydration.
            /// Current-batch LLEP transport belongs to explicit graph stages
            /// and deliberately does not bind its collective to this stage.
            ILocalTPContext *prefill_llep_tp_ctx = nullptr;

            /// Optional graph-owned transfer backing for prefix rehydration.
            /// Current-batch plan/apply stages borrow the same physical slot
            /// directory directly rather than routing it through expert compute.
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
             * Rehydration still uses a dedicated transfer stream. Current-batch
             * movement now runs as explicit same-stream graph phases around a
             * sidebanded model collective and therefore owns no event pair here.
             */
            std::shared_ptr<DeviceMoERebalanceTransferState>
                prefix_runtime_rehydration_transfer_state;
            std::string prefill_llep_workspace_name;

            /**
             * @brief Sole authority for routed-expert weight descriptors.
             *
             * Runtime routing always consumes device-owned top-k ids and
             * weights. This typed value independently selects whether every
             * decode and grouped-prefill projection reads the immutable
             * prepared table or the request-pinned runtime placement bank.
             * Graph lowering chooses it once; setup, capture, and replay must
             * carry it unchanged rather than reconstructing policy from flags.
             */
            MoEDecodeDescriptorSource weight_descriptor_source =
                MoEDecodeDescriptorSource::StaticDescriptorTable;

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

        /**
         * @brief One allocation-free invocation binding for a sparse overlay endpoint.
         *
         * Heterogeneous ExpertOverlay compacts each input row once and preserves
         * that participant's selected experts in original router order along
         * the top-k axis. The enclosing participant stage owns the tensors and
         * immutable residency bank; this structure only lends their stable
         * addresses and fixed-size prepared-engine tables to a persistent nested
         * executor.
         *
         * A serial CPU packet may select another setup-owned compact tensor
         * family and change its live row count, live route width, or residency
         * epoch between transactions. GPU replay families retain one captured
         * tensor identity. The binding never changes model geometry, device,
         * layer, or vector capacities established at construction.
         */
        /** @brief Lifecycle authority represented by a sparse engine binding. */
        enum class SparseOverlayBindingKind : uint8_t
        {
            ResidencyPublication = 0, ///< Live non-zero overlay authority epoch.
            SetupPriming,             ///< One pre-capture construction binding.
            /**
             * Serial CPU packet publication with mutable live tensor geometry.
             *
             * The enclosing local-expert stage remains the sole packet and
             * residency authority. This binding lets its persistent CPU
             * executor reuse kernels and grow-only scratch across packets;
             * GPU captured executors must never select it.
             */
            HostPacketPublication,
        };

        /**
         * @brief Borrowed CPU hidden-row storage for one synchronous host packet.
         *
         * The sparse collective already owns an authenticated FP32 activation
         * matrix. A CPU endpoint must retain its compact row mapping, but it
         * need not copy the complete hidden payload into another tensor before
         * quantization or floating GEMM gather. The enclosing local-expert
         * stage publishes this binding immediately before synchronous CPU
         * execution; GPU and retained asynchronous invocations must leave it
         * empty.
         */
        struct CPUTransportedHiddenRows
        {
            const float *source = nullptr; ///< Stable FP32 transport allocation.
            int source_row_capacity = 0;   ///< Addressable rows in @ref source.
            /** Compact execution row to physical source-row index. */
            std::span<const int> compact_to_source_row;

            /** @return True when no borrowed transport identity was supplied. */
            [[nodiscard]] bool empty() const noexcept
            {
                return source == nullptr && source_row_capacity == 0 &&
                       compact_to_source_row.empty();
            }
        };

        struct SparseOverlayInvocation
        {
            TensorBase *input = nullptr;            ///< Compact FP32 hidden rows.
            TensorBase *routing_indices = nullptr;  ///< Row-major local top-k expert ids.
            TensorBase *routing_weights = nullptr;  ///< Row-major local top-k weights.
            TensorBase *output = nullptr;           ///< Locally aggregated expert rows.
            int live_rows = 0;                      ///< Positive live compact row count.
            /**
             * @brief Live compact route stride for a host packet.
             *
             * Zero retains the construction width. A positive value is legal
             * only for @ref SparseOverlayBindingKind::HostPacketPublication
             * and cannot exceed the setup-certified route capacity.
             */
            int route_width = 0;
            SparseOverlayBindingKind binding_kind =
                SparseOverlayBindingKind::ResidencyPublication;
            /**
             * Exact semantic phase of this retained sparse invocation.
             *
             * HostPacketPublication must select a production phase. Retained
             * participant-local GPU graphs may select their immutable phase at
             * construction. Mapped follower graphs leave this as Auto because
             * their authenticated device-side graph role changes between
             * replays and owns phase identity independently of this binding.
             */
            MoEOverlayServicePhaseHint service_phase =
                MoEOverlayServicePhaseHint::Auto;
            /**
             * Parent residency publication represented by the mask and three
             * prepared-engine tables below. A GPU
             * @ref SparseOverlayBindingKind::ResidencyPublication is immutable
             * for one non-zero generation. A serial CPU packet revalidates its
             * exact tables on every binding because current-batch LLEP may lend
             * transient engines underneath the same durable parent epoch.
             * Setup priming is the sole zero-generation binding.
             */
            uint64_t engine_binding_generation = 0;
            const std::vector<bool> *expert_mask = nullptr; ///< Exact immutable bank mask.
            std::span<ITensorGemm *const> gate_engines;     ///< Global-id-indexed gate engines.
            std::span<ITensorGemm *const> up_engines;       ///< Global-id-indexed up engines.
            std::span<ITensorGemm *const> down_engines;     ///< Global-id-indexed down engines.
            /** Exact CPU-only hidden rows borrowed for this host packet. */
            CPUTransportedHiddenRows cpu_transported_hidden;
        };

        /**
         * @brief Rebind a persistent executor to one serial sparse-overlay packet.
         *
         * This is a manual heterogeneous-boundary API, not a mutable captured
         * graph API. The method copies into construction-sized tables and
         * invalidates device descriptor tables only when prepared engine
         * addresses actually changed. A CPU host-packet binding may select a
         * different preallocated tensor family because it is never captured;
         * a GPU publication may not. It performs no allocation when the
         * constructor established a valid expert geometry.
         *
         * @param invocation Exact compact tensors, live rows, mask, and engines.
         * @return True when the complete invocation was accepted.
         */
        bool bindSparseOverlayInvocation(
            const SparseOverlayInvocation &invocation);

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

        /**
         * @brief Test whether an ownership publication would change this stage.
         *
         * CPU dynamic ownership updates use this predicate to avoid releasing,
         * rebuilding, and rebinding every model layer when a migration wave
         * changes only a sparse set of exact `(layer, expert)` owners.
         *
         * @param candidate Exact expert residency mask proposed for this layer.
         * @return `true` when the stage already owns the candidate mask.
         */
        bool expertMaskMatches(const std::vector<bool> &candidate) const noexcept;

        MoEDecodeDescriptorSource runtimeDecodeDescriptorSourceForTesting() const
        {
            return params_.weight_descriptor_source;
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
        /**
         * @brief Report whether this stage publishes original router slots.
         * @return `true` when a following collective owns route transport.
         *
         * This diagnostic exists so graph-construction tests can distinguish
         * the ownership-invariant publication contract from the obsolete
         * participant-local compact accumulation contract without exposing a
         * mutable parameter object.
         */
        bool publishesCanonicalRouteContributionsForTesting() const noexcept
        {
            return params_.canonical_route_contributions != nullptr;
        }
        /**
         * @brief Return the arithmetic represented by each published route.
         * @return Immutable graph-selected canonical route arithmetic policy.
         */
        MoECanonicalRouteArithmeticPolicy
        canonicalRouteArithmeticPolicyForTesting() const noexcept
        {
            return params_.canonical_route_arithmetic;
        }
        /** @brief Return the immutable canonical publication wire layout. */
        MoECanonicalRoutePublicationLayout
        canonicalRoutePublicationLayoutForTesting() const noexcept
        {
            return params_.canonical_route_layout;
        }
        bool usesRuntimeRowGroupingForTesting() const
        {
            return params_.use_runtime_row_grouping;
        }
        bool supportsRequestedRoutedAssignmentPolicyForTesting() const
        {
            return supportsRequestedRoutedAssignmentPolicy();
        }
        bool hasMoERuntimeTableForTesting() const { return params_.moe_runtime_table != nullptr; }
        /** @return Whether this stage owns one complete observation capability. */
        bool hasOverlayServiceTelemetryForTesting() const noexcept
        {
            return overlay_service_runtime_layer_ &&
                   overlay_service_telemetry_layer_ &&
                   overlay_service_sample_;
        }
        /** @return Typed service phase that the next telemetry marker publishes. */
        MoEOverlayServicePhaseHint
        serviceTelemetryPhaseHintForTesting() const noexcept
        {
            return serviceTelemetryPhaseHint();
        }
        /**
         * @brief Expose the immutable runtime-table binding to graph tests.
         *
         * Production lowering tests use this read-only pointer to prove that a
         * current-batch LLEP stage owns a distinct request-local child while
         * sharing the canonical ExpertOverlay ticket and placement source.
         * Runtime code must not use this diagnostic accessor for publication.
         */
        const IMoERuntimeTable *moeRuntimeTableForTesting() const noexcept
        {
            return params_.moe_runtime_table;
        }
        bool hasPrefillLLEPTPContextForTesting() const { return params_.prefill_llep_tp_ctx != nullptr; }
        bool usesGraphPhasedCurrentBatchLLEPForTesting() const
        {
            return usesGraphPhasedCurrentBatchPrefillLLEP();
        }
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
            for (int expert = 0;
                 expert < normalized_replicas.base_ownership.expertCount();
                 ++expert)
            {
                const bool is_owner =
                    normalized_replicas.ownerParticipant(
                        params_.layer_idx, expert) == socket_id;
                const bool is_replica =
                    normalized_replicas.hasReplicaOnParticipant(
                        params_.layer_idx,
                        expert,
                        socket_id);
                if ((!is_owner && !is_replica) ||
                    (static_cast<size_t>(expert) < params_.expert_mask.size() &&
                     params_.expert_mask[static_cast<size_t>(expert)]))
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

        /**
         * @brief Detach final packed projections for direct CPU ownership movement.
         *
         * The source engines become weightless. The returned native allocations
         * are sent directly into final destination storage without wire blobs.
         */
        ExpertPackedWeights detachPreparedExpert(int expert_id);

        /**
         * @brief Clone final packed projections for direct CPU replica movement.
         *
         * The source engines remain valid while the returned explicit copies
         * become the destination participant's final packed allocations.
         */
        ExpertPackedWeights clonePreparedExpert(int expert_id) const;

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
            const std::unordered_map<int, ExpertWeightBlobs> *received_weights,
            const std::unordered_map<int, PreparedExpertEngines> *
                received_prepared_experts = nullptr);

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
            if (!(supportsGraphCaptureAfterLaunchPreparation() ||
                  params_.prefix_runtime_device_rehydration))
            {
                return GraphLaunchPreparationPolicy::None;
            }
            return sparse_overlay_invocation_bound_ &&
                           sparse_overlay_launch_metadata_dirty_
                       ? GraphLaunchPreparationPolicy::CaptureAndReplay
                       : GraphLaunchPreparationPolicy::CaptureOnly;
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
            grouped_gateup_desc_table_weight_format_ =
                DeviceMoEWeightFormat::NativeVNNI;
            grouped_gateup_desc_table_dirty_ = false;
            grouped_down_desc_table_id_ = -1;
            grouped_down_desc_table_num_experts_ = 0;
            grouped_down_desc_table_d_model_ = 0;
            grouped_down_desc_table_intermediate_ = 0;
            grouped_down_desc_table_weight_format_ =
                DeviceMoEWeightFormat::NativeVNNI;
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
            sparse_overlay_launch_metadata_dirty_ =
                sparse_overlay_invocation_bound_;
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
        /**
         * @brief Declare routed-expert scratch for the complete row envelope.
         *
         * The planner-supplied @p m may cover more rows than this concrete
         * stage instance because one stable workspace backs every serial
         * prefill bucket, decode graph, and grouped verifier graph.  The stage
         * sizes both its MoE-owned buffers and every nested GEMM workspace for
         * the maximum of that family envelope and its concrete `seq_len`.
         *
         * @param m Maximum rows declared by the graph-family planner.
         * @param n Optional output-width hint forwarded to prepared GEMMs.
         * @param k Optional input-width hint forwarded to prepared GEMMs.
         * @return Complete routed-expert and projection workspace requirements.
         */
        WorkspaceRequirements getWorkspaceRequirements(
            int m,
            int n = 0,
            int k = 0) const override;
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;
        void unbindWorkspace() override;
        bool hasWorkspace() const override;
        DeviceWorkspaceManager *getWorkspace() const override;

        /** @inheritdoc IMoEGroupedVerifierHistogramPublisher */
        [[nodiscard]] MoEGroupedVerifierHistogramRole
        groupedVerifierHistogramRole() const noexcept override
        {
            return params_.device_id.is_gpu()
                       ? params_.grouped_verifier_histogram_role
                       : MoEGroupedVerifierHistogramRole::NotOwner;
        }

        /** @inheritdoc IMoEGroupedVerifierHistogramPublisher */
        [[nodiscard]] int
        groupedVerifierHistogramLayerIndex() const noexcept override
        {
            return params_.layer_idx;
        }

        /** @inheritdoc IMoEGroupedVerifierHistogramPublisher */
        [[nodiscard]] std::string_view
        groupedVerifierHistogramPublisherName() const noexcept override
        {
            return "moe_ffn";
        }

        /** @inheritdoc IMoEGroupedVerifierHistogramPublisher */
        [[nodiscard]] void *
        groupedVerifierHistogramPublicationStream() const override
        {
            return params_.moe_runtime_table
                       ? params_.moe_runtime_table
                             ->groupedVerifierHistogramPublicationStream()
                       : nullptr;
        }

        /** @inheritdoc IMoEGroupedVerifierHistogramPublisher */
        bool prepareGroupedVerifierHistogramProducer(
            void *producer_stream) override;

        /** @inheritdoc IMoEGroupedVerifierHistogramPublisher */
        bool transitionGroupedVerifierHistogramProducerCapture(
            void *producer_stream,
            RuntimeHistogramProducerCaptureTransition transition) override;

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
            void *producer_stream) override;

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
         * initializeMoERuntimeTableForGroupedRows(), which validates every
         * persistent device pointer and capacity before returning true.
         */
        void setRuntimePrefillGroupingAvailableForTesting(bool available)
        {
            moe_runtime_row_grouping_available_ = available;
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
         * @brief Query the already-selected weight-descriptor authority.
         * @return True only for request-pinned runtime placement descriptors.
         */
        [[nodiscard]] bool usesRuntimePlacementWeightDescriptors() const noexcept
        {
            return params_.weight_descriptor_source ==
                   MoEDecodeDescriptorSource::RuntimePlacementTable;
        }

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

        /**
         * @brief Name the semantic owner of one CPU grouped-row invocation.
         *
         * Decode, ordinary prefill, and MTP verification share the same
         * economical serial-row-equivalent expert kernels for transported CPU
         * packets. They differ in observability and, outside an already
         * owner-filtered host packet, replica assignment. Carrying that
         * distinction as a scoped enum prevents the executor from inferring
         * lifecycle policy from a mutable test flag or from runtime M.
         */
        enum class CPUGroupedRowsPurpose : uint8_t
        {
            Decode,
            OrdinaryPrefill,
            MTPVerifier,
        };

        /**
         * @brief One locally owned route slot in original row/top-k order.
         *
         * Expert execution is reordered into contiguous expert batches for
         * economy.  This record retains the logical slot identity so the final
         * weighted accumulation can replay the original top-k order exactly,
         * preserving serial-decode FP32 parenthesization byte-for-byte.
         */
        struct CPUGroupedRouteSlot
        {
            int row = 0;       ///< Source/output row in the runtime batch.
            int route = 0;     ///< Original top-k position within @ref row.
            int expert_id = 0; ///< Routed expert owning this work item.
            float weight = 0.0f; ///< Normalized router weight.
        };

        Params params_;
        bool raw_weights_released_ = false;                       ///< Set by releaseRawExpertWeights()
        DeviceWorkspaceManager *bound_workspace_ = nullptr;       ///< Workspace for expert GEMM engines
        ExpertWeightPayloadProvider *payload_provider_ = nullptr; ///< Model-context owned

        /// Cached GEMM engines per expert (resolved on first execute)
        mutable std::vector<ITensorGemm *> cached_gate_gemm_;
        mutable std::vector<ITensorGemm *> cached_up_gemm_;
        mutable std::vector<ITensorGemm *> cached_down_gemm_;

        /**
         * @brief Reusable host scratch for CPU routed-expert execution.
         *
         * Fixed-M verifier stages bind their complete bounded capacity during
         * construction. Heterogeneous CPU ExpertOverlay stages instead borrow
         * @ref CPUGroupedMoESerialWorkspace across mutually exclusive layers
         * and graph roles. Ordinary standalone large-prefill stages retain
         * these private grow-only buffers because they have no serial overlay
         * ownership contract to authorize cross-layer aliasing.
         */
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

        /** @brief Flat locally-owned route slots in original row/top-k order. */
        mutable std::vector<CPUGroupedRouteSlot> cpu_grouped_route_slots_;

        /** @brief Original flat route slot to local-slot index, or -1 when remote. */
        mutable std::vector<int> cpu_grouped_original_to_local_;

        /** @brief Number of locally owned route rows assigned to each expert. */
        mutable std::vector<int> cpu_grouped_expert_counts_;

        /** @brief CSR offsets delimiting each expert's flat local-slot span. */
        mutable std::vector<int> cpu_grouped_expert_offsets_;

        /** @brief Reusable CSR construction cursors. */
        mutable std::vector<int> cpu_grouped_expert_write_offsets_;

        /** @brief Local-slot indices grouped by expert while retaining slot order. */
        mutable std::vector<int> cpu_grouped_expert_local_slots_;

        /**
         * @brief Local logical slot to its expert-major output row.
         *
         * NativeVNNI computes every active expert into one contiguous
         * expert-major layer buffer. This inverse map lets the final
         * row-owned accumulation restore original router order without an
         * intermediate output copy.
         */
        mutable std::vector<int> cpu_grouped_local_to_expert_position_;

        /** @brief Ascending IDs of experts with at least one local route row. */
        mutable std::vector<int> cpu_grouped_active_experts_;

        /** @brief Gather indices reused for each complete expert batch. */
        mutable std::vector<int> cpu_grouped_token_indices_;

        /** Borrowed FP32 transport base for the current synchronous CPU packet. */
        const float *sparse_overlay_cpu_hidden_source_ = nullptr;

        /** Number of addressable rows under @ref sparse_overlay_cpu_hidden_source_. */
        int sparse_overlay_cpu_hidden_source_row_capacity_ = 0;

        /** Compact CPU execution row to borrowed physical source-row index. */
        std::vector<int> sparse_overlay_cpu_compact_to_source_row_;

        /** @brief Canonical router-Q8 rows gathered in expert-major layer order. */
        mutable std::vector<Q8_1Block> cpu_grouped_router_q8_;

        /** @brief Pre-quantized SwiGLU rows for the batched expert down phase. */
        mutable std::vector<Q8_1Block> cpu_grouped_swiglu_q8_;

        /** @brief Floating-format expert outputs indexed by local logical route slot. */
        mutable std::vector<float> cpu_grouped_route_outputs_;

        /** @brief Per-expert route counts used only while PerfStats is enabled. */
        mutable std::vector<uint32_t> cpu_grouped_routed_rows_by_expert_;

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
         * @brief True after routed single-token fused decode has staged pointer arrays.
         *
         * CUDA and ROCm fused MoE decode read device-side pointer tables for the
         * gate/up and down scratch slots during graph replay. Runtime-table and
         * explicit-routing decode share those immutable arrays; only their
         * device-owned source of expert ids and weights differs. Preparation is
         * performed outside capture because the pointer-table publication is a
         * host-to-device metadata upload. Keep this flag tied to the current
         * workspace and descriptor tables so capture can only begin after the
         * exact selected route has been prepared.
         */
        mutable bool runtime_grouped_decode_launch_state_prepared_ = false;

        /// Explicit publication state for the backend-owned fixed-topology mask.
        FixedTopologyMaskPublicationState fixed_topology_mask_publication_state_ =
            FixedTopologyMaskPublicationState::NotRequired;
        /**
         * @brief Whether a host-authoritative heterogeneous boundary rebinds placement.
         *
         * Such a stage retains one native graph but may receive a new immutable
         * residency bank between launches. Graph launch preparation therefore
         * republishes only dirty descriptor/mask bytes before every replay;
         * ordinary homogeneous graphs keep their capture-only preparation.
         */
        bool sparse_overlay_invocation_bound_ = false;
        /** @brief Typed authority that supplied the currently bound invocation. */
        SparseOverlayBindingKind sparse_overlay_binding_kind_ =
            SparseOverlayBindingKind::SetupPriming;
        /** @brief Semantic phase authenticated by the current sparse binding. */
        MoEOverlayServicePhaseHint sparse_overlay_service_phase_ =
            MoEOverlayServicePhaseHint::Auto;
        /** Maximum compact route width certified before any host rebind. */
        int sparse_overlay_route_width_capacity_ = 0;
        /** @brief Residency generation currently certified by the bound tables. */
        uint64_t sparse_overlay_engine_binding_generation_ = 0;
        /**
         * @brief Whether the next sparse replay must republish placement metadata.
         *
         * A retained endpoint executable embeds stable tensor addresses and
         * consumes backend-owned descriptor/mask tables. Revalidating and
         * rebuilding those tables for every unchanged packet is both
         * allocation-heavy and semantically misleading. Binding a different
         * engine or mask sets this bit; successful launch preparation clears
         * it. The policy method above therefore exposes replay preparation
         * only while an actual publication is pending.
         */
        bool sparse_overlay_launch_metadata_dirty_ = false;

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
         * numerically equivalent to ordinary one-token decode. M=1 therefore
         * enters the normal decode route, while every positive multi-row M
         * uses grouped implementations that preserve per-row projection math
         * and top-k accumulation order. This function refuses unsupported GPU
         * requests rather than hiding them behind row replay.
         */
        bool executeDecodeEquivalentVerifierPrefill(IDeviceContext *ctx);

        /**
         * @brief Execute all CPU multi-row routed work through grouped exact kernels.
         *
         * The executor builds a flat CSR schedule without per-expert heap
         * objects, consumes the router's canonical Q8 publication for every
         * NativeVNNI expert bundle, and invokes one runtime-M gate/up and down
         * operation per active expert.  Expert outputs remain separate until
         * they are accumulated in original row/top-k order, so ordinary
         * prefill and MTP verification are both serial-row byte-equivalent.
         *
         * @param ctx Active CPU device context.
         * @param purpose Lifecycle policy selecting ordinary-prefill ownership
         *                or verifier replica assignment and PerfStats labels.
         * @return true after every local route slot has been computed and
         *         accumulated; false on any invalid or unsupported contract.
         */
        bool executeCPUGroupedDecodeEquivalentRows(
            IDeviceContext *ctx,
            CPUGroupedRowsPurpose purpose);

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
         * @brief Validate the cold explicit-routing one-token capture contract.
         *
         * Heterogeneous ExpertOverlay followers receive router ids and weights
         * directly in participant-local device tensors. They intentionally have
         * no host mirror or MoE runtime table. This predicate admits that typed
         * route only when fused mask-aware execution has every stable tensor and
         * local expert engine it needs; it does not claim that pointer arrays or
         * the immutable ownership mask have already been published.
         */
        bool supportsExplicitRoutingDecodeGraphCapturePreflight() const;

        /**
         * @brief Return whether explicit-routing fused decode is capture-ready.
         *
         * @return true after descriptor tables, scratch pointer arrays, and any
         * participant ownership mask have been published on the exact graph
         * stream.
         */
        bool isExplicitRoutingDecodeGraphCapturable() const;

        /**
         * @brief Publish graph-stable state for explicit-routing fused decode.
         *
         * This operation runs only before capture. It builds sparse global
         * descriptor tables with entries for participant-local experts, uploads
         * the immutable local ownership mask, and stages the backend's reusable
         * gate/up and down pointer arrays. No routing values are read on the host.
         */
        bool prepareExplicitRoutingDecodeLaunchState(IMoEKernel *kernel);
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
        bool initializeMoERuntimeTableForGroupedRows();
        bool initializeFixedTopologyGroupedPrefill();
        int expectedGroupedDecodeParticipantCount() const;
        bool runtimeTableHasActiveGroupedDecodeBank() const;
        bool supportsRequestedRoutedAssignmentPolicy() const;
        bool canUseRuntimeRowGrouping() const;
        /**
         * @brief Return whether this captured stage publishes ordinary prefill demand.
         *
         * Deferred grouped-verifier rows are published by the accepted-state
         * transaction instead. Keeping the two producer roles disjoint makes
         * stream admission complete without double-registering semantics.
         */
        [[nodiscard]] bool ownsRuntimePrefillHistogramPublication() const noexcept;
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
        bool usesGraphPhasedCurrentBatchPrefillLLEP() const noexcept;
        bool hasValidCompactLLEPTransferBinding() const;
        bool executePrefixRuntimeRehydrationPayloadMovement(
            IMoEKernel *kernel,
            DeviceMoERebalanceStatus **transfer_status_out,
            DeviceMoERebalanceApplyStatus **apply_status_out,
            DeviceMoERebalanceTransferState *transfer_state_override =
                nullptr) const;
        /**
         * @brief Validate the immutable one-row runtime-table launch contract.
         *
         * The same production kernel serves ordinary decode and an exact-shape
         * one-row prefill left by prefix restore. This predicate checks only
         * immutable geometry and bindings; graph preparation owns descriptor
         * publication and runtime-bank readiness.
         */
        bool supportsDeviceRoutedOneRowGraphCapturePreflight() const;
        /** @brief Check prepared launch state for the one-row runtime-table path. */
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
        /**
         * @brief Execute the routed path between device timing markers.
         *
         * The public @ref execute method owns the optional Dynamic service
         * markers. Keeping the existing implementation behind this helper
         * guarantees that every successful decode, prefill, verifier, and
         * sparse-participant exit reaches one common finish publication.
         */
        bool executeWithoutServiceTelemetry(IDeviceContext *ctx);

        /**
         * @brief Resolve the sole semantic phase for service accounting.
         *
         * A packet-bound sparse phase has precedence over the immutable graph
         * phase. Auto then permits ordinary graph geometry or the device-owned
         * runtime role to supply identity at execution.
         */
        MoEOverlayServicePhaseHint serviceTelemetryPhaseHint() const noexcept;
        const DeviceMoEPlacementBank *activeRuntimePlacementBank() const;
        bool runtimeLocalComputeEnabled(const DeviceMoEPlacementBank *bank, int expert_id) const;
        void ensureScratchBuffers(int max_batch) const;
        IMoEKernel *ensureMoEKernel() const;

        mutable int grouped_gateup_desc_table_id_ = -1;
        mutable int grouped_gateup_desc_table_num_experts_ = 0;
        mutable int grouped_gateup_desc_table_d_model_ = 0;
        mutable int grouped_gateup_desc_table_intermediate_ = 0;
        mutable DeviceMoEWeightFormat grouped_gateup_desc_table_weight_format_ =
            DeviceMoEWeightFormat::NativeVNNI;
        bool grouped_gateup_desc_table_dirty_ = false;

        mutable int grouped_down_desc_table_id_ = -1;
        mutable int grouped_down_desc_table_num_experts_ = 0;
        mutable int grouped_down_desc_table_d_model_ = 0;
        mutable int grouped_down_desc_table_intermediate_ = 0;
        mutable DeviceMoEWeightFormat grouped_down_desc_table_weight_format_ =
            DeviceMoEWeightFormat::NativeVNNI;
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
        /** Route-count source used only by the service observation kernel. */
        DeviceMoELayerRuntime *overlay_service_runtime_layer_ = nullptr;
        /** First of this layer's three canonical device-local service cells. */
        DeviceMoEOverlayServiceTelemetryCell *overlay_service_telemetry_layer_ =
            nullptr;
        /** Serial retained-graph timing cursor owned by the runtime table. */
        DeviceMoEOverlayServiceTelemetrySample *overlay_service_sample_ =
            nullptr;
        bool moe_runtime_table_initialized_ = false;
        bool moe_runtime_row_grouping_available_ = false;
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
             * @brief Optional binding for required router-owned GPU Q8 rows.
             *
             * This object exposes only immutable capture-time row addresses;
             * the shared expert retains a private MoE kernel and private
             * grouping scratch. When non-null, execution requires the exact
             * router publication and fails hard if its source or geometry does
             * not match. When null, the grouped verifier owns standalone input
             * quantization; deterministic mode uses that branch because router
             * Q8 reuse is intentionally disabled there.
             */
            std::shared_ptr<MoERouterQ8HiddenPublication>
                required_router_q8_publication;
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
        /** @brief Test whether grouped execution requires a router Q8 publication. */
        bool requiresRouterQ8PublicationForTesting() const
        {
            return params_.required_router_q8_publication != nullptr;
        }

        // =====================================================================
        // IWorkspaceConsumer Implementation
        // =====================================================================
        /**
         * @brief Declare shared-expert scratch for the complete row envelope.
         *
         * Grouped shared-expert execution owns MoE metadata in addition to its
         * three GEMM workspaces.  All of those buffers use the maximum of the
         * concrete stage rows and the planner's serial-family rows so captured
         * graph addresses remain valid across request-shape changes.
         *
         * @param m Maximum rows declared by the graph-family planner.
         * @param n Optional model-width hint for prepared GEMMs.
         * @param k Optional intermediate-width hint for prepared GEMMs.
         * @return Complete shared-expert workspace requirements.
         */
        WorkspaceRequirements getWorkspaceRequirements(
            int m,
            int n = 0,
            int k = 0) const override;
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

        /**
         * @brief Resolve the three immutable prepared shared-expert GEMM engines.
         *
         * Resolution is allocation-free and may happen before a GPU stream is
         * assigned. The engines receive the stage's exact stream immediately
         * before launch preparation or arithmetic execution.
         */
        void ensureGemmEnginesCached() const;
        /**
         * @brief Publish the one-entry typed gate/up table used by grouped decode.
         *
         * The prepared gate, up, and down engines are validated as one uniform
         * NativeVNNI or FP16/BF16/FP32 family before the backend receives any
         * descriptor. The backend owns the resulting graph-stable table.
         *
         * @param kernel Exact-stream-bound backend MoE kernel.
         * @param d_model Gate/up reduction width.
         * @param intermediate Gate/up output width.
         * @return True when the immutable table is available for capture.
         */
        bool ensureSharedGroupedGateUpDescriptorTable(IMoEKernel *kernel, int d_model, int intermediate) const;
        /**
         * @brief Publish the one-entry typed down table used by grouped decode.
         *
         * @param kernel Exact-stream-bound backend MoE kernel.
         * @param d_model Down-projection output width.
         * @param intermediate Down-projection reduction width.
         * @return True when the immutable table is available for capture.
         */
        bool ensureSharedGroupedDownDescriptorTable(IMoEKernel *kernel, int d_model, int intermediate) const;
        /** @return Whether this stage is a GPU grouped-verifier prefill launch. */
        bool shouldUseGroupedVerifierPrefillRoute() const;
        /** @return Whether verifier rows must use the serial decode oracle. */
        bool shouldUseDecodeEquivalentVerifierPrefill() const;
        /** @return Whether ordinary single-row execution uses grouped tables. */
        bool shouldUseGroupedDecodeRoute() const;
        /** @brief Execute the grouped verifier-prefill route when selected. */
        bool tryGroupedVerifierPrefill(IMoEKernel *kernel, int d_model, int intermediate) const;
        /** @brief Execute the grouped single-row route when selected. */
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
        /**
         * @brief Immutable ownership of the shared-gate arithmetic.
         *
         * A normal all-reduce leaves a complete row on every participant, so
         * each graph owns its local gate. A rooted reduction leaves valid bytes
         * only on its fixed root; the sibling node must then remain a capture-
         * symmetric no-op until the root broadcasts the final combined row.
         */
        enum class ExecutionRole
        {
            ReplicatedOwner, ///< Every graph owns a complete shared row.
            RootOwner,       ///< This graph owns the rooted reduction result.
            NonRootObserver  ///< Symmetric graph node with no tensor authority.
        };

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

            /** Typed arithmetic authority selected by graph lowering. */
            ExecutionRole execution_role = ExecutionRole::ReplicatedOwner;

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
        CoherencePolicy coherencePolicy() const override
        {
            return params_.execution_role == ExecutionRole::NonRootObserver
                       ? CoherencePolicy::NONE
                       : CoherencePolicy::FULL;
        }
        StageDumpInfo buildDumpInfoImpl() const override;
        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;
        void unbindWorkspace() override;
        bool hasWorkspace() const override;
        DeviceWorkspaceManager *getWorkspace() const override;

        /**
         * @return Immutable graph-local arithmetic authority selected during lowering.
         *
         * Graph validators use this typed view to prove that a non-root
         * participant's symmetric stage cannot read or write root-owned
         * shared/routed tensors.
         */
        [[nodiscard]] ExecutionRole executionRole() const noexcept
        {
            return params_.execution_role;
        }

        /**
         * @brief Reset the private backend object after a hard graph reset.
         */
        void invalidateKernelDynamicState() override;

    private:
        Params params_;

        /** @return Whether this graph owns and may mutate the shared row. */
        [[nodiscard]] bool ownsGateArithmetic() const noexcept
        {
            return params_.execution_role != ExecutionRole::NonRootObserver;
        }

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
     * @brief Graph-local ownership role for canonical route reduction.
     *
     * The reducer needs to know whether this participant owns arithmetic; it
     * does not need communicator indices. Keeping that decision typed avoids
     * reconstructing ownership from two raw integers and lets a single-device
     * graph declare local ownership without inventing a collective root.
     */
    enum class MoECanonicalRouteReductionRole
    {
        Unspecified,
        RootOwner,
        NonRootParticipant,

        /** Every participant owns reduction after an allreduce of route slots. */
        AllreduceParticipant,
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
            /** Router weights required by unweighted CPU canonical slots. */
            TensorBase *routing_weights = nullptr;
            TensorBase *output = nullptr;
            int seq_len = 0;
            int top_k = 0;
            int d_model = 0;
            /** Exact arithmetic encoded by canonical route slots. */
            MoECanonicalRouteArithmeticPolicy canonical_route_arithmetic =
                MoECanonicalRouteArithmeticPolicy::Unspecified;
            /** Dense GPU slots or packed indexed CPU records. */
            MoECanonicalRoutePublicationLayout canonical_route_layout =
                MoECanonicalRoutePublicationLayout::Unspecified;
            /** Explicit graph-local arithmetic ownership; never inferred. */
            MoECanonicalRouteReductionRole reduction_role =
                MoECanonicalRouteReductionRole::Unspecified;
            /**
             * Optional production sparse fabric replacing a dense LocalTP
             * route-slot reduction. Presence makes every continuation
             * participant active: peers publish assigned slots and the root
             * acquires/folds them in original router order.
             */
            std::shared_ptr<MoEOverlayNodeLocalRouteExchange>
                node_local_route_exchange;
            /**
             * Final device-owned assignment within the continuation domain.
             * A `-1` slot belongs to another overlay domain and is completed by
             * the heterogeneous return path. The reducer never reconstructs
             * this invocation-local schedule from setup metadata.
             */
            MoEDomainRouteAssignmentLedger domain_route_assignment{};
            /** Typed source for routes assigned outside this continuation domain. */
            MoEExternalCanonicalRouteSource external_route_source =
                MoEExternalCanonicalRouteSource::Unspecified;
            /** Workload-typed final weight publication consumed by execution. */
            MoERuntimeRouteWeightBinding runtime_route_weights{};
            /**
             * Request-pinned overlay-wide expert placement authority.
             *
             * Production packet dispatch consumes this same two-bank binding.
             * The reducer exposes non-owning diagnostic views only, allowing
             * parity evidence to distinguish global tier placement from the
             * domain-local route schedule without creating a host shadow.
             */
            MoEOverlayRoutePlacementDeviceBinding overlay_route_placement{};
            /** Stable domain-local participant represented by this graph. */
            int route_participant_id = -1;
            BufferId canonical_route_contributions_buffer_id =
                BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS;
            BufferId routing_weights_buffer_id = BufferId::MOE_EXPERT_WEIGHTS;
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
        /**
         * @brief Certify the statically non-owning rooted-reduction role.
         *
         * A `NonRootParticipant` never reads, writes, publishes, or launches a
         * reducer; it waits for the following rooted broadcast. This immutable
         * role can therefore join the root's reduction capture wave passively
         * without recording an empty backend graph.
         */
        bool isPassiveGraphCaptureNoOp() const override
        {
            return !params_.node_local_route_exchange &&
                   params_.reduction_role ==
                   MoECanonicalRouteReductionRole::NonRootParticipant;
        }
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
         * @brief Declare coherence on every participant that runs arithmetic.
         *
         * A rooted GPU transaction executes arithmetic only on `RootOwner`;
         * its non-root nodes are symmetric declarative no-ops. A CPU route-slot
         * allreduce leaves a complete slot tensor on every participant, so each
         * `AllreduceParticipant` performs the ordered FMA fold and owns normal
         * input/output coherence as well.
         */
        CoherencePolicy coherencePolicy() const override
        {
            return params_.node_local_route_exchange ||
                           params_.reduction_role ==
                           MoECanonicalRouteReductionRole::RootOwner ||
                           params_.reduction_role ==
                               MoECanonicalRouteReductionRole::
                                   AllreduceParticipant
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
        /**
         * Root-owned record lookup allocated once with graph construction.
         * Entry `i` is the gathered record containing original route slot `i`.
         */
        std::vector<int32_t> packed_record_for_route_slot_;
        /** Root-device copy of endpoint-correct mapped peer descriptors. */
        std::shared_ptr<MoEOverlayPersistentGraphStorage>
            node_local_peer_bindings_storage_;
        /** Root-device semantic status shared by acquire/validate/fold nodes. */
        std::shared_ptr<MoEOverlayPersistentGraphStorage>
            node_local_validation_storage_;
        /** Capture-stable producer alias for a non-root participant. */
        MoENodeLocalRoutePeerDeviceBinding node_local_producer_binding_{};
        /** Number of descriptors in root storage; zero on producer graphs. */
        std::uint32_t node_local_peer_count_ = 0u;
        /**
         * Shared typed view of the final domain schedule and pinned placement.
         *
         * The same object is used by mapped packet producers in topologies
         * without this reducer, so diagnostic ownership follows the real route
         * authority instead of depending on LocalTP cardinality.
         */
        std::unique_ptr<MoEOverlayPinnedRouteEvidenceViews>
            pinned_route_evidence_views_;
        mutable std::unique_ptr<IMoEKernel> owned_moe_kernel_;
        mutable IMoEKernel *moe_kernel_ = nullptr;
    };

    /**
     * @brief Gather sparse indexed CPU route rows to one fixed TP root.
     *
     * The preceding expert stage writes allocation-free packed records into the
     * canonical publication tensor. This stage gathers only those active
     * records through the cross-rank CPU TP communicator and publishes the
     * gathered record count in the same fixed trailer. Non-root participants
     * retain their local bytes but do not read them after this node.
     */
    class MoECanonicalRouteGatherStage final : public IComputeStage
    {
    public:
        /** @brief Immutable graph-bound packed-gather parameters. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            ITPContext *tp_ctx = nullptr;
            TensorBase *packed_route_records = nullptr;
            int seq_len = 0;
            int top_k = 0;
            int d_model = 0;
            int root_participant = 0;
            std::string stage_name;
            BufferId packed_route_records_buffer_id =
                BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS;
        };

        explicit MoECanonicalRouteGatherStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override
        {
            return ComputeStageType::ROOTED_COLLECTIVE;
        }
        std::string name() const override
        {
            return "moe_canonical_route_gather";
        }
        size_t estimatedFlops() const override { return 0u; }
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override { return false; }
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;
        CoherencePolicy coherencePolicy() const override
        {
            return CoherencePolicy::FULL;
        }
        StageDumpInfo buildDumpInfoImpl() const override;

        /** @brief Return immutable graph parameters for focused tests. */
        [[nodiscard]] const Params &params() const noexcept { return params_; }

    private:
        Params params_;
    };

    /**
     * @brief Broadcast the compact canonical CPU MoE result from its root.
     *
     * Only the exact active `[seq_len, d_model]` prefix is transported. The
     * stage follows the root-only ordered reducer explicitly in the graph, so
     * every participant consumes the same ownership-invariant bytes without
     * broadcasting bucket-capacity tail storage.
     */
    class MoECanonicalOutputBroadcastStage final : public IComputeStage
    {
    public:
        /** @brief Immutable graph-bound compact-broadcast parameters. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            ITPContext *tp_ctx = nullptr;
            TensorBase *output = nullptr;
            int seq_len = 0;
            int d_model = 0;
            int root_participant = 0;
            std::string stage_name;
            BufferId output_buffer_id = BufferId::MOE_COMBINED_OUTPUT;
        };

        explicit MoECanonicalOutputBroadcastStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override
        {
            return ComputeStageType::ROOTED_COLLECTIVE;
        }
        std::string name() const override
        {
            return "moe_canonical_output_broadcast";
        }
        size_t estimatedFlops() const override { return 0u; }
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override { return false; }
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;
        CoherencePolicy coherencePolicy() const override
        {
            return CoherencePolicy::FULL;
        }
        StageDumpInfo buildDumpInfoImpl() const override;

        /** @brief Return immutable graph parameters for focused tests. */
        [[nodiscard]] const Params &params() const noexcept { return params_; }

    private:
        Params params_;
    };

    /**
     * @brief Publish a participant-local shared FFN row into a canonical bank.
     *
     * This stage is the device-resident join between independently runnable
     * routed and shared branches. The routed expert owns the route-slot prefix
     * of the canonical publication tensor; this stage preserves that prefix and
     * overwrites every participant-bank element in the suffix. Exactly one bank
     * contains the local shared partial and all peer banks contain zero, so the
     * following rooted sum transports values without selecting their arithmetic
     * reduction order.
     *
     * The operation is intentionally a first-class graph node. Its dependencies
     * make both branch producers visible, its inout buffer contract prevents
     * stale route-prefix coherence, and its immutable participant identity is
     * captured with the graph. No host publication, allocation, or dynamic
     * launch decision is permitted.
     */
    class MoESharedExpertRankBankPublishStage final : public IComputeStage
    {
    public:
        /** @brief Immutable graph-bound rank-bank publication parameters. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            TensorBase *shared_output = nullptr;
            TensorBase *canonical_publication = nullptr;
            int seq_len = 0;
            int top_k = 0;
            int d_model = 0;
            int participant_device_index = -1;
            int participant_count = 0;
            /** Device-owned live row count for padded captured graphs. */
            const int32_t *active_row_count_device = nullptr;
            BufferId shared_output_buffer_id =
                BufferId::MOE_SHARED_EXPERT_OUTPUT;
            BufferId canonical_publication_buffer_id =
                BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS;
        };

        explicit MoESharedExpertRankBankPublishStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override
        {
            return ComputeStageType::MOE_SHARED_RANK_BANK_PUBLISH;
        }
        std::string name() const override
        {
            return "moe_shared_rank_bank_publish";
        }
        size_t estimatedFlops() const override { return 0; }
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override;
        bool supportsGraphCaptureAfterLaunchPreparation() const override;
        bool supportsLazyPrefillGraphCapturePreflight() const override;
        bool supportsPaddedPrefillGraphCapturePreflight() const override;
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
        {
            return supportsGraphCaptureAfterLaunchPreparation()
                       ? GraphLaunchPreparationPolicy::CaptureOnly
                       : GraphLaunchPreparationPolicy::None;
        }
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

        /** @brief Expose immutable graph policy for structural tests. */
        [[nodiscard]] const Params &params() const noexcept { return params_; }

        /** @brief Inject a non-owning kernel oracle in device-free stage tests. */
        void setMoEKernelForTesting(IMoEKernel *kernel)
        {
            owned_moe_kernel_.reset();
            moe_kernel_ = kernel;
        }

    private:
        Params params_;
        mutable std::unique_ptr<IMoEKernel> owned_moe_kernel_;
        mutable IMoEKernel *moe_kernel_ = nullptr;
    };

    /**
     * @brief Finalize a rooted canonical MoE transaction in fixed FP32 order.
     *
     * Every participant owns this graph node to keep LocalTP graphs symmetric,
     * but only the fixed collective root accesses tensors or launches a kernel.
     * The root folds route slots in router order, folds shared banks in ascending
     * participant order, evaluates the established shared gate reduction, and
     * writes routed, gated-shared, and final combined rows in one launch. The
     * following rooted broadcast publishes only the final combined row.
     *
     * Root ownership is part of immutable graph policy. A non-root node has an
     * empty coherence contract and cannot accidentally validate or publish stale
     * local output; a root node requires all inputs and outputs explicitly.
     */
    class MoECanonicalPublicationFinalizeStage final : public IComputeStage
    {
    public:
        /** @brief Immutable graph-bound finalization parameters. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            TensorBase *input = nullptr;
            TensorBase *gate_inp = nullptr;
            TensorBase *canonical_publication = nullptr;
            TensorBase *routed_output = nullptr;
            TensorBase *shared_output = nullptr;
            TensorBase *combined_output = nullptr;
            int seq_len = 0;
            int top_k = 0;
            int d_model = 0;
            int participant_device_index = -1;
            int root_device_index = -1;
            int participant_count = 0;
            /** Device-owned live row count for padded captured graphs. */
            const int32_t *active_row_count_device = nullptr;
            BufferId input_buffer_id = BufferId::NORMALIZED;
            BufferId canonical_publication_buffer_id =
                BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS;
            BufferId routed_output_buffer_id =
                BufferId::MOE_COMBINED_OUTPUT;
            BufferId shared_output_buffer_id =
                BufferId::MOE_SHARED_EXPERT_OUTPUT;
            BufferId combined_output_buffer_id = BufferId::ATTN_PROJ;
        };

        explicit MoECanonicalPublicationFinalizeStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override
        {
            return ComputeStageType::MOE_CANONICAL_PUBLICATION_FINALIZE;
        }
        std::string name() const override
        {
            return "moe_canonical_publication_finalize";
        }
        size_t estimatedFlops() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override;
        bool supportsGraphCaptureAfterLaunchPreparation() const override;
        bool supportsLazyPrefillGraphCapturePreflight() const override;
        bool supportsPaddedPrefillGraphCapturePreflight() const override;
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
        {
            return supportsGraphCaptureAfterLaunchPreparation()
                       ? GraphLaunchPreparationPolicy::CaptureOnly
                       : GraphLaunchPreparationPolicy::None;
        }
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;
        CoherencePolicy coherencePolicy() const override
        {
            return params_.participant_device_index ==
                           params_.root_device_index
                       ? CoherencePolicy::FULL
                       : CoherencePolicy::NONE;
        }
        StageDumpInfo buildDumpInfoImpl() const override;

        /** @brief Expose immutable graph policy for structural tests. */
        [[nodiscard]] const Params &params() const noexcept { return params_; }

        /** @brief Inject a non-owning kernel oracle in device-free stage tests. */
        void setMoEKernelForTesting(IMoEKernel *kernel)
        {
            owned_moe_kernel_.reset();
            moe_kernel_ = kernel;
        }

    private:
        Params params_;
        mutable std::unique_ptr<IMoEKernel> owned_moe_kernel_;
        mutable IMoEKernel *moe_kernel_ = nullptr;
    };

} // namespace llaminar2
