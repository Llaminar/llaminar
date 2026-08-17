/**
 * @file MoELocalExpertStage.h
 * @brief Participant-local sparse-MoE compute and serial graph-family buffers.
 *
 * A heterogeneous ExpertOverlay graph transports a compact set of routed rows
 * to the participant that owns each expert.  @ref MoELocalExpertStage turns
 * that host-visible packet into the prepared local GEMM invocation and returns
 * the compact result to the explicit sparse collective.  The file also defines
 * @ref MoELocalExpertSerialBufferArena: a typed, immutable-address owner for
 * the transient compact tensors shared by graph roles that the orchestrator
 * has proven cannot execute concurrently. Each arena may contain a small
 * decode/MTP family and a maximum prefill family with independent coherence;
 * this avoids capacity-wide decode transfers while retaining an explicit
 * ownership boundary for concurrent graph families. Production overlay
 * arenas retain a bounded power-of-two ladder through one captured prefill
 * segment, keeping transfer and fixed-launch padding below two times the live
 * route count without sizing transient state from the full KV context.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../moe/MoEOverlaySparseCollective.h"
#include "../../moe/MoEOverlayParticipantResidency.h"
#include "../../moe/MoEExpertWeightService.h"
#include "../../moe/CPUCurrentBatchLLEP.h"
#include "../../moe/MoERuntimeTable.h"
#include "../../../loaders/ExpertSlabTypes.h"

#include <cstddef>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{
    class FP32Tensor;
    class TensorBase;
    class ITensorGemm;
    class PreparedWeightStore;
    class ExpertGemmRegistry;
    class ExpertWeightPayloadProvider;
    class DeviceWorkspaceManager;
    class MoEExpertComputeStage;
    class IWorkerGPUContext;
    class PinnedHostTransferBuffer;

    /**
     * @brief Immutable compact-route tensor owner for one serial graph family.
     *
     * The owner is intentionally narrower than a general tensor cache.  A
     * graph builder creates it only after it has selected one participant and
     * one device whose graph roles are ordered by explicit graph/event edges.
     * Every captured or manual-boundary stage receives stable addresses from
     * one of the arena's immutable capacity families. A caller that can run
     * two graph roles concurrently must create separate arenas; the type
     * exposes no resize operation that could invalidate a live graph identity.
     */
    class MoELocalExpertSerialBufferArena final
    {
    public:
        /** @brief Immutable capacity and layout contract for one arena. */
        struct Config
        {
            DeviceId device_id = DeviceId::invalid();
            /** @brief Largest compact input-row capacity admitted by this arena. */
            size_t row_capacity = 0;
            /**
             * @brief Optional immutable capacities prepared beside the maximum.
             *
             * Each value describes a complete, separately coherent tensor
             * family. The arena always adds @ref row_capacity, sorts the
             * result, and removes duplicates. A heterogeneous endpoint can
             * therefore select a live-size family without treating a prefix
             * of a larger tensor as independently coherent.
             */
            std::vector<size_t> row_capacity_buckets;
            int d_model = 0;
            int routing_top_k = 0;
            /**
             * @brief Optional logical sparse participant that exclusively owns this arena.
             *
             * Graph-native ExpertOverlay construction supplies this value so
             * a local stage can reject accidental cross-participant aliasing
             * even when two participants happen to use the same physical
             * device.  Direct, non-overlay construction leaves it empty.
             */
            std::optional<int> logical_participant_id;
            std::string debug_name;
        };

        /**
         * @brief One independently coherent compact tensor family.
         *
         * All four tensor addresses are immutable for the arena lifetime.
         * Families are ordered by ascending row capacity and are selected
         * only at the explicit heterogeneous manual boundary.
         */
        struct TensorFamily
        {
            /** @brief Maximum compact input rows stored by this family. */
            size_t row_capacity = 0;
            /** @brief Sum of the four tensor payload sizes. */
            size_t allocation_bytes = 0;
            /** @brief Complete backend-pinned H2D/D2H family bytes, zero on CPU. */
            size_t pinned_transfer_bytes = 0;
            /** @brief Hidden-row source offset in @ref pinned_transfer. */
            size_t pinned_hidden_offset = 0;
            /** @brief Routing-index source offset in @ref pinned_transfer. */
            size_t pinned_routing_indices_offset = 0;
            /** @brief Routing-weight source offset in @ref pinned_transfer. */
            size_t pinned_routing_weights_offset = 0;
            /** @brief Expert-output destination offset in @ref pinned_transfer. */
            size_t pinned_output_offset = 0;
            /** @brief Compacted hidden rows. */
            std::shared_ptr<FP32Tensor> hidden;
            /** @brief Original-order local top-k expert ids encoded as FP32. */
            std::shared_ptr<FP32Tensor> routing_indices;
            /** @brief Original-order local top-k routing weights. */
            std::shared_ptr<FP32Tensor> routing_weights;
            /** @brief Compact participant-local expert outputs. */
            std::shared_ptr<FP32Tensor> output;
            /**
             * @brief Stable host transfer family embedded in retained GPU graphs.
             *
             * The owner contains compact hidden/index/weight H2D sources and
             * the compact output D2H destination.  It binds one exact CUDA/HIP
             * allocation during model setup. All layer graphs for this
             * participant may share it because the arena contract proves those
             * graphs execute serially.
             */
            std::shared_ptr<PinnedHostTransferBuffer> pinned_transfer;
        };

        /**
         * @brief Allocate all compact host tensors at their final graph-family capacity.
         *
         * Device storage remains placement-owned and is created through
         * TransferEngine before a GPU graph captures a consumer.  The tensor
         * objects themselves, their host allocations, and any resulting device
         * addresses remain stable for this arena's lifetime.
         *
         * @throws std::invalid_argument when the immutable geometry is invalid.
         * @throws std::overflow_error when the tensor geometry cannot be represented.
         */
        explicit MoELocalExpertSerialBufferArena(Config config);

        /**
         * @brief Build compact power-of-two families through an exact upper bound.
         *
         * Decode and grouped-verifier packets commonly leave only one or two
         * rows on a participant even though the admitted prefill envelope is
         * much larger. Preparing these small immutable families during
         * setup avoids transferring and executing the complete envelope for
         * every sparse endpoint. The exact @p maximum_capacity is always the
         * final element, including when it is not a power of two.
         *
         * @param maximum_capacity Largest compact row count in the family.
         * @return Strictly increasing positive capacities, or an empty vector
         *         when @p maximum_capacity is zero.
         */
        static std::vector<size_t> powerOfTwoRowBucketsThrough(
            size_t maximum_capacity);

        /**
         * @brief Select the smallest power-of-two route width that fits a row.
         *
         * The final model top-k is always an exact bucket, including when it
         * is not a power of two. This scalar helper is allocation-free so the
         * sparse packet boundary can choose an already-captured graph family
         * without changing tensor addresses or launch topology.
         *
         * @param required_routes Largest participant-local route count in one row.
         * @param model_top_k Maximum routing width admitted by the model.
         * @return A value in `[required_routes, model_top_k]`, or zero for an
         *         invalid/empty request.
         */
        static int routeWidthBucketFor(
            int required_routes,
            int model_top_k) noexcept;

        MoELocalExpertSerialBufferArena(
            const MoELocalExpertSerialBufferArena &) = delete;
        MoELocalExpertSerialBufferArena &operator=(
            const MoELocalExpertSerialBufferArena &) = delete;
        MoELocalExpertSerialBufferArena(
            MoELocalExpertSerialBufferArena &&) = delete;
        MoELocalExpertSerialBufferArena &operator=(
            MoELocalExpertSerialBufferArena &&) = delete;

        /** @brief Return the exact participant-local execution device. */
        const DeviceId &deviceId() const noexcept { return device_id_; }
        /** @brief Return the maximum number of compact input rows. */
        size_t rowCapacity() const noexcept { return row_capacity_; }
        /** @brief Return the hidden/output feature width. */
        int dModel() const noexcept { return d_model_; }
        /** @brief Return the compact routing width used by local GEMM. */
        int routingTopK() const noexcept { return routing_top_k_; }
        /** @brief Return the optional logical participant bound at construction. */
        const std::optional<int> &logicalParticipantId() const noexcept
        {
            return logical_participant_id_;
        }
        /** @brief Return the sum of the four host/device tensor payload sizes. */
        size_t allocationBytes() const noexcept { return allocation_bytes_; }
        /** @brief Return setup-owned pinned bytes used by captured GPU transfers. */
        size_t pinnedTransferBytes() const noexcept
        {
            return pinned_transfer_bytes_;
        }
        /** @brief Return immutable tensor families in ascending capacity order. */
        const std::vector<TensorFamily> &families() const noexcept
        {
            return families_;
        }
        /** @brief Return the number of setup-owned capacity families. */
        size_t familyCount() const noexcept { return families_.size(); }

        /**
         * @brief Find the smallest independently coherent family that fits a packet.
         * @param requested_row_capacity Number of compact rows required now.
         * @return Matching family, or nullptr when the request exceeds admission.
         */
        const TensorFamily *smallestFamilySupporting(
            size_t requested_row_capacity) const noexcept;

        /**
         * @brief Check whether a stage can safely bind this immutable arena.
         *
         * The requested row count may be smaller than the fixed capacity,
         * but every device and tensor-layout property must match exactly.
         */
        bool supports(
            DeviceId device,
            size_t requested_row_capacity,
            int d_model,
            int routing_top_k) const noexcept;

        /** @brief Return the shared compact hidden-row tensor. */
        const std::shared_ptr<FP32Tensor> &hidden() const noexcept
        {
            return families_.back().hidden;
        }
        /** @brief Return the shared row-major local top-k index tensor. */
        const std::shared_ptr<FP32Tensor> &routingIndices() const noexcept
        {
            return families_.back().routing_indices;
        }
        /** @brief Return the shared row-major local top-k weight tensor. */
        const std::shared_ptr<FP32Tensor> &routingWeights() const noexcept
        {
            return families_.back().routing_weights;
        }
        /** @brief Return the shared compact local-expert output tensor. */
        const std::shared_ptr<FP32Tensor> &output() const noexcept
        {
            return families_.back().output;
        }

    private:
        DeviceId device_id_ = DeviceId::invalid();
        size_t row_capacity_ = 0;
        int d_model_ = 0;
        int routing_top_k_ = 0;
        std::optional<int> logical_participant_id_;
        size_t allocation_bytes_ = 0;
        size_t pinned_transfer_bytes_ = 0;
        std::vector<TensorFamily> families_;
    };

    /**
     * @brief Execute one participant-local sparse-MoE expert packet.
     *
     * The stage owns no routing or peer coordination.  Its input and output
     * views are supplied by the graph's sparse collective stages, prepared
     * expert engines are resolved before construction, and all GPU ownership
     * is expressed through TransferEngine plus the executor's exact stream.
     */
    class MoELocalExpertStage : public IComputeStage,
                                public IWorkspaceConsumer,
                                public ICPUCurrentBatchLLEPExpertConsumer
    {
    public:
        /**
         * @brief Authority permitted to resolve participant-local expert GEMMs.
         *
         * Tiered ExpertOverlay graphs use @ref RegistryOnly: the prepared
         * registry is the sole authority for an exact expert slice, so a
         * missing registry entry cannot silently fall back to an unrelated raw
         * three-dimensional parent binding. Hand-built and legacy local graphs
         * retain @ref AllowRawFallback while they materialize their own views.
         */
        enum class ExpertWeightResolutionPolicy
        {
            AllowRawFallback,
            RegistryOnly,
        };

        /**
         * @brief Select who materializes a submitted participant-local result.
         *
         * CPU endpoints and standalone stages finish in their producer node.
         * A heterogeneous rank containing several GPUs uses
         * @ref DeferredExplicitStage so every device receives its work before
         * any host-visible output wait begins.
         */
        enum class CompletionPolicy : uint8_t
        {
            Inline = 0,             ///< Submit and finish inside execute().
            DeferredExplicitStage, ///< Require one graph-owned completion node.
        };

        /** @brief Immutable construction inputs for one participant-local stage. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            const MoEOverlaySparseRows *input_rows = nullptr;
            std::shared_ptr<const MoEOverlaySparseRows> input_rows_lifetime;
            MoEOverlayReturnRows *output_rows = nullptr;
            std::shared_ptr<MoEOverlayReturnRows> output_rows_lifetime;
            std::shared_ptr<MoEOverlayCollectiveWorkspace> workspace_lifetime;
            /**
             * Immutable compact tensors shared by one ordered graph family.
             *
             * The graph builder supplies this only for serially scheduled
             * variants.  An absent arena deliberately keeps stage-private
             * ownership for hand-built or concurrent graph instances.
             */
            std::shared_ptr<MoELocalExpertSerialBufferArena>
                serial_compact_buffer_arena;
            /**
             * Exact compact input-row capacity of this graph variant.
             *
             * Zero derives the capacity from the packet view. A serial-family
             * packet view may be admitted for maximum prefill rows while a
             * decode graph consumes only one fixed row; that graph must set a
             * positive value so construction and GPU preparation select its
             * smaller independently coherent tensor bucket.
             */
            size_t graph_row_capacity = 0;
            /**
             * @brief Explicit completion ownership for this local endpoint.
             *
             * Deferred completion is valid only for a GPU-backed,
             * epoch-indexed ExpertOverlay participant. The graph must insert a
             * matching @ref MoELocalExpertCompletionStage before any return
             * transport consumes @ref output_rows.
             */
            CompletionPolicy completion_policy = CompletionPolicy::Inline;
            /**
             * @brief Exact GPU resource authority for retained endpoint graphs.
             *
             * A deferred heterogeneous endpoint launches a participant-local
             * captured graph on the exact context-owned participant stream.
             * The cache borrows that stream so compact H2D preparation, graph
             * replay, and output publication form one ordered queue without
             * per-layer handoff events. CPU and inline endpoints leave the
             * pointer null.
             */
            IWorkerGPUContext *worker_gpu_context = nullptr;

            TensorBase *gate_exps = nullptr;
            TensorBase *up_exps = nullptr;
            TensorBase *down_exps = nullptr;
            int num_experts = 0;
            int top_k = 0;
            int d_model = 0;
            int expert_intermediate = 0;
            int layer_idx = -1;
            std::vector<bool> expert_mask;

            /**
             * Participant-local epoch-indexed prepared-engine authority.
             *
             * Production ExpertOverlay graphs bind this object so a packet
             * executes the exact bank named by its residency epoch. The owner
             * outlives every cached graph; the stage never caches a mutable
             * current-bank pointer.
             */
            std::shared_ptr<MoEOverlayParticipantResidency>
                overlay_participant_residency;
            /**
             * Optional graph-owned CPU current-batch child transaction.
             *
             * The begin and restore stages bracket this exact local endpoint.
             * While active, sparse packets retain the durable parent epoch but
             * execute through the transient resident mask and destination map.
             */
            std::shared_ptr<CPUCurrentBatchLLEPTransactionState>
                cpu_current_batch_llep_state;

            // ================================================================
            // Prepared expert state — analogous to MoEExpertComputeStage::Params.
            // No peer participant / domain runtime / runner fields. The optional
            // MoE runtime table below is the graph-facing placement table only.
            // When prepared_gate_gemm is non-empty (size == num_experts), the
            // execute() path skips inline extractExpertViews /
            // prepareExpertGemmEngines and uses these engines directly.
            // ================================================================
            std::vector<ITensorGemm *> prepared_gate_gemm;
            std::vector<ITensorGemm *> prepared_up_gemm;
            std::vector<ITensorGemm *> prepared_down_gemm;
            /// Retains extracted per-expert views used by non-store engines.
            std::vector<std::shared_ptr<TensorBase>> expert_gate_views;
            std::vector<std::shared_ptr<TensorBase>> expert_up_views;
            std::vector<std::shared_ptr<TensorBase>> expert_down_views;
            /// Keeps MoE batch-constructed kernel objects alive alongside the stage.
            std::vector<std::shared_ptr<ITensorGemm>> moe_owned_kernels;
            /// Retains backend-native packed slabs when ownership is stage-local.
            std::shared_ptr<void> moe_packed_gate_lifetime;
            std::shared_ptr<void> moe_packed_up_lifetime;
            std::shared_ptr<void> moe_packed_down_lifetime;
            /// Pointer to model-context-owned PreparedWeightStore.  Not owned here.
            PreparedWeightStore *prepared_store = nullptr;
            /// Pointer to model-context-owned ExpertGemmRegistry.  Not owned here.
            ExpertGemmRegistry *expert_registry = nullptr;
            /**
             * Resolution authority enforced while the graph is materialized.
             * ExpertOverlay must remain registry-only so its raw parent tensors
             * can never become an accidental alternate execution path.
             */
            ExpertWeightResolutionPolicy expert_weight_resolution_policy =
                ExpertWeightResolutionPolicy::AllowRawFallback;
            /// Stable graph-facing MoE runtime placement state for this overlay/local participant.
            /// Owned by the graph/model layer; stages only cache the per-layer pointer.
            IMoERuntimeTable *moe_runtime_table = nullptr;
            /// Optional logical participant id recorded in runtime placement descriptors.
            int runtime_participant_index = -1;
            /**
             * Exact producer stream for cold GPU runtime-table publication.
             *
             * CPU stages leave this null. A GPU stage that needs to initialize
             * its placement bank fails immediately when no stream was supplied.
             */
            void *runtime_publication_stream = nullptr;
            /// Cached slab references for PreparedWeightStore-based resolution.
            std::optional<ExpertSlabRef> gate_slab_ref;
            std::optional<ExpertSlabRef> up_slab_ref;
            std::optional<ExpertSlabRef> down_slab_ref;
        };

        static_assert(StageParamsRequired<Params>);

        /**
         * @brief Resolve every participant-local expert engine at graph build.
         *
         * Production execution never extracts expert views or constructs GEMM
         * engines. Graph builders must call this method after assigning the
         * participant mask and before moving @p params into a stage.
         *
         * Existing complete registry-provided tables are retained. Otherwise
         * the shared expert-weight service prepares the active engines and
         * publishes stable slab references/lifetimes into @p params.
         *
         * @return true when every active expert has gate, up, and down engines.
         */
        static bool prepareExpertGemmEngines(Params &params);

        /**
         * @brief Bind immutable packet views, prepared weights, and compact storage.
         * @throws std::runtime_error when the supplied fixed-capacity arena cannot
         *         serve the sparse packet geometry.
         */
        explicit MoELocalExpertStage(Params params);

        /** @brief Destroy only after no deferred invocation remains in flight. */
        ~MoELocalExpertStage() override;

        /** @brief Execute one addressed sparse packet on the participant's device. */
        bool execute(IDeviceContext *ctx) override;
        /**
         * @brief Materialize and fold the one previously submitted GPU packet.
         *
         * The call is legal exactly once after a deferred execute(). It retains
         * the selected residency bank, persistent nested kernel owner, compact
         * tensors, and route schedule until the output publication has crossed
         * the heterogeneous host boundary.
         *
         * @param ctx Exact participant GPU context used by the submit stage.
         * @return True after the return rows and completion evidence are ready.
         */
        bool completeDeferredOutput(IDeviceContext *ctx);
        /** @return Whether this stage requires an explicit completion node. */
        bool usesDeferredCompletion() const noexcept
        {
            return params_.completion_policy ==
                   CompletionPolicy::DeferredExplicitStage;
        }
        /** @return Whether submit has produced work not yet consumed by completion. */
        bool hasPendingDeferredOutput() const noexcept;
        /** @brief Validate that every locally active expert has prepared weights. */
        bool validatePreparedWeights(std::string *error) const override;
        /**
         * @brief Retain setup validation when an immutable residency authority owns engines.
         *
         * installReadyBank() exhaustively validates each expert triplet before
         * publishing an immutable shared bank. Live packets then acquire and
         * validate the exact epoch/device/layer bank before touching an engine,
         * so rescanning the construction vectors cannot strengthen that proof.
         */
        PreparedWeightValidationLifetime
        preparedWeightValidationLifetime() const noexcept override
        {
            return params_.overlay_participant_residency
                       ? PreparedWeightValidationLifetime::StageLifetime
                       : PreparedWeightValidationLifetime::PerExecution;
        }
        ComputeStageType type() const override { return ComputeStageType::MOE_LOCAL_EXPERT; }
        std::string name() const override { return "moe_local_expert"; }
        size_t estimatedFlops() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override { return false; }
        /** @return true because ExpertOverlay local expert work is an explicit heterogeneous replay unit. */
        bool isManualGraphBoundary() const override { return true; }
        bool supportsPaddedPrefillGraphCapturePreflight() const override
        {
            return true;
        }
        bool supportsPaddedPrefillRealLengthContract() const override
        {
            return true;
        }
        bool allowsZeroOutput() const override { return true; }
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }
        StageBufferRequirements getBufferRequirements() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

        /** @brief Advertise nested grouped-GEMM workspace required by this endpoint. */
        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;
        /** @brief Bind graph-owned grouped-GEMM workspace before GPU execution. */
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;
        /** @brief Release the borrowed grouped-GEMM workspace binding. */
        void unbindWorkspace() override;
        bool hasWorkspace() const override { return bound_workspace_ != nullptr; }
        DeviceWorkspaceManager *getWorkspace() const override { return bound_workspace_; }

        /** @brief Return the immutable stage construction contract for diagnostics. */
        const Params &params() const { return params_; }
        /**
         * @brief Return the compact hidden tensor identity used by this stage.
         *
         * The returned pointer is read-only diagnostic evidence.  It lets graph
         * ownership tests prove that serial graph variants share the intended
         * arena without exposing mutable storage or a resize operation.
         */
        const FP32Tensor *compactHiddenTensorForDiagnostics() const noexcept
        {
            return compact_hidden_.get();
        }
        /** @return Currently bound independently coherent compact-row capacity. */
        size_t compactRowCapacityForDiagnostics() const noexcept
        {
            return compact_capacity_;
        }
        /** @return Transformer layer whose participant-local experts this stage owns. */
        int layerIndex() const noexcept { return params_.layer_idx; }
        /** @return Stable overlay participant identity used by the owner map. */
        int participantIndex() const noexcept
        {
            return params_.runtime_participant_index;
        }

        /** @copydoc ICPUCurrentBatchLLEPExpertConsumer::cpuCurrentBatchLLEPLayerIndex */
        int cpuCurrentBatchLLEPLayerIndex() const noexcept override
        {
            return layerIndex();
        }

        /** @copydoc ICPUCurrentBatchLLEPExpertConsumer::cpuCurrentBatchLLEPParticipantId */
        int cpuCurrentBatchLLEPParticipantId() const noexcept override
        {
            return participantIndex();
        }

        /** @copydoc ICPUCurrentBatchLLEPExpertConsumer::cpuCurrentBatchLLEPUsesCPU */
        bool cpuCurrentBatchLLEPUsesCPU() const noexcept override
        {
            return params_.device_id.is_cpu();
        }

        /** @copydoc ICPUCurrentBatchLLEPExpertConsumer::cloneCPUCurrentBatchLLEPPreparedExpert */
        ExpertPackedWeights cloneCPUCurrentBatchLLEPPreparedExpert(
            int expert_id,
            uint64_t durable_parent_epoch) const override;

        /** @copydoc ICPUCurrentBatchLLEPExpertConsumer::installCPUCurrentBatchLLEPTransientResidency */
        bool installCPUCurrentBatchLLEPTransientResidency(
            const CPUCurrentBatchLLEPTransactionState &state,
            const std::unordered_map<int, PreparedExpertEngines> *arrivals)
            override;

        /** @copydoc ICPUCurrentBatchLLEPExpertConsumer::discardCPUCurrentBatchLLEPTransientResidency */
        bool discardCPUCurrentBatchLLEPTransientResidency(
            const CPUCurrentBatchLLEPTransactionState &state) noexcept override;
        /** @return Whether the currently published resident mask equals @p candidate. */
        bool expertMaskMatches(const std::vector<bool> &candidate) const noexcept
        {
            return params_.expert_mask == candidate;
        }
        /** @return Current participant-local resident expert mask. */
        const std::vector<bool> &expertMask() const noexcept
        {
            return params_.expert_mask;
        }

        /** @brief Serialize one resident prepared expert for cross-backend transport. */
        ExpertWeightBlobs serializeExpert(int expert_id) const;

        /** @brief Return requested experts that do not have all three prepared projections. */
        std::vector<int> missingPreparedExpertIds(
            const std::vector<int> &expert_ids) const;

        /**
         * @brief Retire prepared engines excluded by @p new_mask.
         *
         * This is the post-publication retirement phase. Callers must serialize
         * or clone every departing expert and publish the replacement residency
         * epoch before invoking it.
         */
        std::vector<const TensorBase *> releaseDepartedExperts(
            const std::vector<bool> &new_mask);

        /**
         * @brief Install already-staged arrivals without changing live routing.
         *
         * The expert mask and runtime bank remain unchanged until
         * @ref applyExpertMask succeeds, so dispatch cannot observe a partially
         * prepared residency wave.
         */
        bool registerAndPrepareNewExperts(
            const std::vector<bool> &new_mask,
            const std::unordered_map<int, ExpertWeightBlobs> *received_weights,
            const std::unordered_map<int, PreparedExpertEngines> *
                received_prepared_experts = nullptr);

        /**
         * @brief Atomically publish the participant's new resident mask.
         *
         * When a runtime table is bound, this builds a complete inactive bank
         * from prepared engines and flips it on the exact publication stream.
         * No graph topology or captured address changes.
         */
        void applyExpertMask(const std::vector<bool> &new_mask);

        /** @brief Build the shared weight-service view over stage-owned state. */
        MoEWeightContext buildWeightContext();

        /** @brief Bind the model-owned immutable payload provider for GPU arrivals. */
        void setPayloadProvider(ExpertWeightPayloadProvider *provider) noexcept
        {
            payload_provider_ = provider;
        }

        /** @brief Refresh the device-visible active placement bank for this layer. */
        bool refreshRuntimePlacement();

        /**
         * @brief Allocate the compact sparse-route tensors before inference.
         *
         * The heterogeneous boundary republishes host rows on every call, but
         * their host and device addresses are model-lifetime state.  GPU
         * callers must bind the exact producer stream before invoking this
         * method; subsequent executions transfer new bytes into the existing
         * allocations and never allocate in the inference path.
         *
         * @return true when the complete physical route capacity is ready.
         */
        bool preparePersistentBuffers();

    private:
        /** One compact `(input row, expert, route weight)` execution record. */
        struct ActiveRoute
        {
            size_t input_row = 0;
            int expert_id = -1;
            float weight = 0.0f;
        };

        /** Lifecycle shared by the submit node and its explicit completion node. */
        enum class DeferredLifecycle : uint8_t
        {
            Idle = 0,   ///< No packet is retained.
            ReadyNoWork, ///< An authenticated empty packet still needs acknowledgement.
            Submitted, ///< GPU work and its exact producer event are in flight.
        };

        /** @brief Bind a private or serial-family compact tensor set at fixed capacity. */
        bool ensureCompactCapacity(size_t rows, int routing_top_k) const;
        /**
         * @brief Execute packet preparation and GPU submission on the current thread.
         *
         * The rank-local graph calls every endpoint submit node before entering
         * its completion wave.  Each retained endpoint graph therefore becomes
         * independently runnable on its exact participant stream without a
         * second host worker transaction around the already-asynchronous GPU
         * launch.
         */
        bool executePacketOnCurrentThread(IDeviceContext *ctx);
        /**
         * @brief Await publication and build return rows on the current thread.
         *
         * The explicit completion wave reaches this only after every sibling
         * endpoint has submitted its retained graph.  Waiting on the exact
         * producer event here preserves multi-device overlap while avoiding
         * redundant thread-pool scheduling around HIP/CUDA graph launches.
         */
        bool completeDeferredOutputOnCurrentThread(IDeviceContext *ctx);
        /**
         * @brief Build retained GPU graphs for every tensor/route-width family.
         *
         * Decode/MTP and maximum-prefill families have different embedded tensor
         * addresses. Each row family also owns fixed 1/2/4/.../top-k launch
         * widths, allowing sparse participants to avoid executing invalid route
         * slots without recapture. Every family remains a complete retained
         * graph with immutable pointer and launch identity.
         */
        void createDeferredReplayFamilies();
        /** @brief Return the retained graph bound to current tensors and route width. */
        struct DeferredGPUReplayFamily;
        DeferredGPUReplayFamily *currentDeferredReplayFamily() noexcept;
        /** @brief Return the deferred lifecycle to Idle and release its epoch bank. */
        void resetDeferredInvocation() noexcept;
        /**
         * @brief Record one complete host-visible endpoint service duration.
         *
         * CPU, CUDA, and ROCm observations use the same packet-to-return-row
         * wall-time boundary. This prevents a kernel-only GPU event from being
         * compared against a materially different CPU cost when the residency
         * planner decides whether an expert migration will accelerate live
         * heterogeneous inference.
         */
        bool publishServiceMeasurement(
            ExpertHistogramSource source,
            uint64_t elapsed_nanoseconds,
            uint64_t activations);
        /** @brief Publish the initial participant-local placement bank when needed. */
        bool initializeMoERuntimePlacementBank();
        /** @brief Build and atomically flip one complete placement bank. */
        bool publishMoERuntimePlacementBank(uint32_t epoch);
        /** @brief Return whether the active runtime bank is valid for this endpoint. */
        bool runtimeTableHasActiveOverlayBank() const;
        /** @brief Return whether a device-visible expert descriptor permits local work. */
        bool runtimeLocalComputeEnabled(int expert_id) const;
        /** @brief Return whether the static participant mask excludes all experts. */
        bool staticExpertMaskDisablesAllExperts() const;
        /** @brief Return whether the active runtime bank admits any input route locally. */
        bool hasRuntimeLocalWorkForInput(const MoEOverlaySparseRows &input) const;
        /** @brief Validate one routed expert against the currently active authority. */
        bool isExpertActiveForValidation(int expert_id) const;
        /** @brief Rebind registry engines after an asynchronous expert arrival. */
        void bindPreparedExpertEnginesForExperts(
            const std::vector<int> &expert_ids);

        Params params_;
        /** Maximum live route and row counts admitted by this graph identity. */
        size_t route_admission_capacity_ = 0;
        size_t row_admission_capacity_ = 0;
        /** Allocation-free packet scratch reused after each explicit completion. */
        std::vector<ActiveRoute> active_routes_;
        std::vector<int> row_output_slot_;
        std::vector<uint8_t> validated_input_rows_;
        std::vector<size_t> output_input_rows_;
        /** Number of original-order local routes packed into each compact row. */
        std::vector<size_t> compact_row_route_counts_;
        std::vector<ITensorGemm *> invocation_gate_engines_;
        std::vector<ITensorGemm *> invocation_up_engines_;
        std::vector<ITensorGemm *> invocation_down_engines_;
        std::vector<bool> invocation_expert_mask_;
        /**
         * @brief Residency epoch represented by the cached invocation tables.
         *
         * Deferred GPU families retain their construction-sized engine arrays
         * across serial packets. A matching non-zero authority epoch proves
         * those arrays and the mask are unchanged; a new epoch rebuilds all
         * four tables before the retained executable is submitted.
         */
        uint64_t invocation_engine_binding_generation_ = 0;
        /**
         * Persistent per-capacity graphs; never destroyed while endpoint work is in flight.
         *
         * The implementation type owns backend graph streams/events and stays
         * private so this public stage header does not expose graph-executor
         * machinery to every compute-stage consumer.
         */
        std::vector<std::unique_ptr<DeferredGPUReplayFamily>>
            deferred_replay_families_;
        /** Exact retained family whose captured publication is currently in flight. */
        DeferredGPUReplayFamily *pending_deferred_replay_ = nullptr;
        DeferredLifecycle deferred_lifecycle_ = DeferredLifecycle::Idle;
        std::shared_ptr<const MoEOverlayParticipantResidencyBank>
            deferred_residency_bank_;
        bool pending_economy_timing_enabled_ = false;
        bool pending_endpoint_detail_enabled_ = false;
        bool pending_profiling_enabled_ = false;
        size_t pending_compact_family_bytes_ = 0;
        std::chrono::steady_clock::time_point pending_service_start_{};
        std::chrono::steady_clock::time_point pending_stage_setup_start_{};
        std::chrono::steady_clock::time_point pending_compute_start_{};
        std::chrono::steady_clock::time_point pending_compute_end_{};
        mutable size_t compact_capacity_ = 0;
        mutable std::shared_ptr<FP32Tensor> compact_hidden_;
        mutable std::shared_ptr<FP32Tensor> compact_routing_indices_;
        mutable std::shared_ptr<FP32Tensor> compact_routing_weights_;
        mutable std::shared_ptr<FP32Tensor> compact_output_;
        /**
         * Full tensor stride owned by the serial arena. This never changes and
         * remains the maximum model top-k allocation contract.
         */
        mutable int compact_routing_top_k_ = 0;
        /**
         * Exact fixed launch width selected for the current packet. Routing
         * entries are repacked with this stride before choosing the matching
         * retained graph; zero means no packet is bound.
         */
        int compact_execution_top_k_ = 0;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;
        DeviceMoELayerRuntime *moe_runtime_layer_ = nullptr;
        bool moe_runtime_table_initialized_ = false;
        ExpertWeightPayloadProvider *payload_provider_ = nullptr;
        /** Fixed expert-indexed prepared replicas owned only by the active child. */
        std::vector<PreparedExpertEngines>
            cpu_current_batch_llep_transient_arrivals_;
        /** Exact durable epoch whose immutable bank supplies owner engines. */
        uint64_t cpu_current_batch_llep_parent_epoch_ = 0;
    };

    /**
     * @brief Explicit graph node that completes one submitted local GPU expert packet.
     *
     * The producer pointer is graph-lifetime stable: ComputeGraph owns both
     * nodes and destroys stages only after execution has quiesced. Construction
     * rejects CPU, inline, or cross-device producers, while execute() enforces
     * the single-submit/single-completion lifecycle.
     */
    class MoELocalExpertCompletionStage final : public IComputeStage
    {
    public:
        /** @brief Immutable producer binding for one graph-owned completion edge. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;
            MoELocalExpertStage *producer = nullptr; ///< Stable graph-owned submit stage.
        };

        static_assert(StageParamsRequired<Params>);

        /**
         * @brief Validate a matching deferred producer and exact GPU device.
         * @throws std::invalid_argument when the graph ownership contract is invalid.
         */
        explicit MoELocalExpertCompletionStage(Params params);

        /** @brief Complete and publish the producer's retained packet. */
        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override
        {
            return ComputeStageType::MOE_LOCAL_EXPERT_COMPLETION;
        }
        std::string name() const override
        {
            return "moe_local_expert_completion";
        }
        /** @return True for CPU/CUDA/ROCm executors; construction requires GPU. */
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override { return false; }
        bool isManualGraphBoundary() const override { return true; }
        bool allowsZeroOutput() const override { return true; }
        CoherencePolicy coherencePolicy() const override
        {
            return CoherencePolicy::NONE;
        }
        /** @brief Describe the paired producer identity for stage diagnostics. */
        StageDumpInfo buildDumpInfoImpl() const override;

        /** @return Immutable producer binding for graph diagnostics. */
        const Params &params() const noexcept { return params_; }

    private:
        Params params_;
    };

} // namespace llaminar2
