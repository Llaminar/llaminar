/**
 * @file MoERoutingStage.h
 * @brief MoE routing stage: softmax top-k expert selection
 *
 * Extracted from MoEExpertComputeStage to enable independent routing computation.
 * Outputs raw routing results (expert indices as float, normalized weights)
 * without any expert-ID ownership masking; MoEExpertComputeStage handles that.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../memory/BufferId.h"
#include "../../../kernels/IMoEKernel.h"
#include "../../moe/DecodeExpertHistogram.h"
#include "../../moe/IMoEGroupedVerifierHistogramPublisher.h"
#include "../../moe/MoERuntimeTable.h"

#include <memory>
#include <string>
#include <vector>
namespace llaminar2
{
    /**
     * @brief Authority that consumes one GPU decode router publication.
     *
     * Ordinary homogeneous decode publishes both top-k tensors and the
     * persistent device runtime table consumed by the local expert kernel.
     * A heterogeneous ExpertOverlay graph instead captures the top-k tensors
     * into a fixed-capacity dispatch ticket; its explicitly declared manual
     * boundary owns placement and expert execution.  Keeping that distinction
     * typed prevents a missing runtime table from silently selecting grouped
     * prefill routing in an unrelated decode graph.
     */
    enum class MoEDecodeRoutePublicationPolicy
    {
        DeviceRuntimeTable,
        FixedCapacityOverlayTicket,
    };


    /**
     * @brief MoE routing stage: compute expert selection via softmax top-k
     *
     * Calls IMoEKernel::route() to compute routing, then writes results
     * to output tensors as FP32. It does not apply expert-ID ownership masking;
     * the responsibility of MoEExpertComputeStage.
     *
     * Outputs:
     * - output_indices: FP32 [seq_len * top_k] expert IDs cast to float
     * - output_weights: FP32 [seq_len * top_k] normalized routing weights
     */
    class MoERoutingStage : public IComputeStage,
                            public IWorkspaceConsumer,
                            public IMoEGroupedVerifierHistogramPublisher,
                            public IMoEHostGroupedVerifierHistogramPublisher
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            // Input
            TensorBase *input = nullptr; ///< Normalized hidden [seq_len, d_model]
            int seq_len = 0;
            int d_model = 0;

            // Router config
            TensorBase *gate_weights = nullptr; ///< Router gate [num_experts, d_model]
            int num_experts = 0;
            int top_k = 0;
            bool norm_topk_prob = true;

            // Layer info for histogram
            int layer_idx = -1;
            DecodeExpertHistogram *decode_histogram = nullptr;
            /**
             * @brief CPU-owned logical rows eligible for routing evidence.
             *
             * CPU graphs have exact host-owned request geometry, but the
             * routing tensor may still have a larger physical row capacity in
             * focused tests or future fixed-width execution.  When a CPU
             * histogram is attached and `seq_len > 1`, graph construction must
             * publish the exact leading row count here.  The stage rejects a
             * missing or out-of-range count instead of allowing physical
             * padding to bias Dynamic expert placement.
             *
             * GPU callers must leave this zero.  Their logical row count is
             * device-owned through `active_row_count_device`, and routing must
             * never download it merely to update a host histogram.
             */
            int host_logical_row_count = 0;
            IMoERuntimeTable *moe_runtime_table = nullptr;
            /**
             * @brief Collect selected/local expert counts in the device runtime table.
             *
             * Observe and Dynamic execution consume these counters when they
             * evaluate placement economy. Static execution has no such
             * consumer, so enabling collection there would add two global
             * atomics per selected route to every captured decode layer for no
             * semantic benefit. Graph lowering owns this decision; callers
             * must not infer it from whether a runtime table happens to exist.
             */
            bool collect_device_runtime_histogram = true;
            /** @brief Typed owner of single-row GPU decode route metadata. */
            MoEDecodeRoutePublicationPolicy decode_route_publication =
                MoEDecodeRoutePublicationPolicy::DeviceRuntimeTable;
            /**
             * @brief Graph-lowered ownership of each selected routed row.
             *
             * This must match the paired MoEExpertComputeStage. A physically
             * replicated placement can either distribute rows for LLEP economy
             * or execute every row independently for a mirrored MTP sidecar;
             * routing owns that distinction because it publishes the runtime
             * expert IDs consumed by the expert kernel.
             */
            RoutedExpertRowExecutionPolicy routed_row_execution_policy =
                RoutedExpertRowExecutionPolicy::ParticipantAssigned;
            bool force_grouped_verifier_prefill_for_decode = false;
            /**
             * @brief Device-owned absolute position row shared with RoPE.
             *
             * Runtime decode uses this stable semantic coordinate when
             * partitioning replicated experts. The pointer is graph-local and
             * remains stable across capture replay while its contents advance
             * on the graph's publication stream.
             */
            const int32_t *absolute_position_ids_device = nullptr;
            /**
             * @brief Explicit graph-local owner shared with the routed expert stage.
             *
             * The paired stages are sequential and exchange router-produced Q8
             * hidden rows through this backend object. A distinct owner must be
             * created for every separately captured main or MTP graph.
             */
            std::shared_ptr<MoERoutedPipelineKernelOwner> routed_pipeline_kernel_owner;

            // Optional graph-capturable decode rebalance apply piggyback.
            // When enabled, the route kernel applies any ready device-side
            // rebalance wave before publishing this token's top-k/local masks.
            bool device_rebalance_route_apply = false;
            std::string device_rebalance_workspace_name;
            DeviceMoERebalanceConfig device_rebalance_config;
            DeviceMoEExpertDirectoryEntry *device_rebalance_local_transfer_slots = nullptr;
            uint32_t device_rebalance_local_transfer_slot_count = 0;
            uint32_t device_rebalance_plan_capacity = 0;
            uint32_t device_rebalance_command_buffer_count = 1;
            int device_rebalance_apply_layer_idx = -2;

            /**
             * @brief Select grouped serial-decode-equivalent verifier routing.
             *
             * MTP all-position verifier batches produce several candidate rows
             * at once.  GPU backends execute those rows as one economical
             * grouped launch whose per-row K traversal and reduction order are
             * byte-equivalent to M=1 decode.  This flag never permits production
             * row replay; it selects the dedicated grouped verifier contract.
             */
            bool force_decode_equivalent_verifier_prefill = false;

            /**
             * @brief Declare this router's grouped-verifier history role.
             *
             * A manual heterogeneous ExpertOverlay graph has no local grouped
             * expert stage on the continuation GPU, so its decode-equivalent
             * router is the last device stage that sees all selected expert IDs.
             * @ref MoEGroupedVerifierHistogramRole::DeferredAcceptedRows makes
             * that router fuse selected-ID retention into top-k publication.
             * Participant IDs are retained as `-1`: global demand drives overlay
             * placement, while local participant demand belongs to the remote
             * execution domains. Static placement selects the same lifecycle
             * boundary with @ref MoEGroupedVerifierHistogramRole::StaticNoPublication,
             * but allocates and publishes no demand history.
             *
             * This is valid only for a GPU main verifier with a complete
             * per-layer runtime ledger.  Ordinary and LocalTP overlay graphs
             * use @ref MoEGroupedVerifierHistogramRole::NotOwner because their
             * expert stage owns the per-layer boundary.
             */
            MoEGroupedVerifierHistogramRole grouped_verifier_histogram_role =
                MoEGroupedVerifierHistogramRole::NotOwner;

            /**
             * @brief Device-owned logical row count for every variable-row GPU graph.
             *
             * Bucketed prefill and grouped verification both launch a stable
             * physical M while only a leading prefix belongs to the current
             * request. The graph captures this model-lifetime address, whose
             * value is published by request admission or verifier preparation
             * on the graph's producer stream. Backends invalidate every suffix
             * route directly from this scalar, so no host replay parameter or
             * scalar upload can influence captured topology.
             */
            const int32_t *active_row_count_device = nullptr;

            // Outputs (written by this stage)
            TensorBase *output_indices = nullptr; ///< FP32 [seq_len * top_k] expert IDs as float
            TensorBase *output_weights = nullptr; ///< FP32 [seq_len * top_k] normalized weights
            // Buffer IDs for coherence
            BufferId input_buffer_id = BufferId::NORMALIZED;
            BufferId output_indices_buffer_id = BufferId::MOE_EXPERT_INDICES;
            BufferId output_weights_buffer_id = BufferId::MOE_EXPERT_WEIGHTS;
        };

        explicit MoERoutingStage(Params params);
        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_ROUTER; }
        std::string name() const override { return "moe_router"; }
        size_t estimatedFlops() const override;

        int layerIndex() const { return params_.layer_idx; }

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
            return "moe_router_overlay_ticket";
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

        /**
         * @brief Commit accepted overlay-verifier selections from this ledger.
         *
         * @copydetails IMoEGroupedVerifierHistogramPublisher::enqueueCommittedGroupedVerifierHistograms
         */
        bool enqueueCommittedGroupedVerifierHistograms(
            const int32_t *accepted_state_counts_device,
            const int32_t *publication_ok_flags_device,
            int request_count,
            int rows_per_request,
            void *producer_stream) override;

        /** @inheritdoc IMoEHostGroupedVerifierHistogramPublisher */
        [[nodiscard]] bool
        requiresHostGroupedVerifierHistogramPublication() const noexcept override
        {
            return params_.device_id.is_cpu() &&
                   params_.decode_histogram != nullptr &&
                   params_.force_decode_equivalent_verifier_prefill;
        }

        /** @inheritdoc IMoEHostGroupedVerifierHistogramPublisher */
        [[nodiscard]] int
        hostGroupedVerifierHistogramLayerIndex() const noexcept override
        {
            return params_.layer_idx;
        }

        /** @inheritdoc IMoEHostGroupedVerifierHistogramPublisher */
        [[nodiscard]] std::string_view
        hostGroupedVerifierHistogramPublisherName() const noexcept override
        {
            return "moe_router_cpu_grouped_verifier";
        }

        /**
         * @brief Validate retained CPU routes against accepted request geometry.
         *
         * @copydetails IMoEHostGroupedVerifierHistogramPublisher::validateHostGroupedVerifierHistogramPublication
         */
        [[nodiscard]] bool validateHostGroupedVerifierHistogramPublication(
            const int32_t *accepted_state_counts,
            int request_count,
            int rows_per_request,
            std::string *error) const override;

        /**
         * @brief Commit accepted CPU verifier prefixes into routing demand.
         *
         * @copydetails IMoEHostGroupedVerifierHistogramPublisher::publishHostGroupedVerifierHistograms
         */
        bool publishHostGroupedVerifierHistograms(
            const int32_t *accepted_state_counts,
            int request_count,
            int rows_per_request,
            std::string *error) override;

        bool allowsZeroOutput() const override { return false; }
        bool isGraphCapturable() const override;
        bool supportsGraphCaptureAfterLaunchPreparation() const override;
        /**
         * @brief Describe every routing graph-capture admission predicate.
         *
         * A router is often the first stage to expose an incomplete
         * device-resident MoE lifecycle. Include both cold-preflight and warm
         * readiness facts so a mandatory full-graph failure identifies the
         * missing contract directly instead of merely reporting MOE_ROUTER.
         */
        std::string graphCaptureReadinessDebugString() const override;
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
        bool supportsBackend(ComputeBackendType backend) const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

        /**
         * @brief Declare routing storage for the complete graph-family row envelope.
         *
         * The concrete stage records the row count of the graph that owns it,
         * while the workspace allocator may pass a larger row count covering
         * every prefill bucket that can reuse the same stable workspace
         * addresses.  The implementation must honor both values.  Sizing only
         * from `params_.seq_len` makes the first captured request determine the
         * lifetime capacity of `moe_route_logits` and the quantized router
         * inputs, so a later larger request cannot safely reuse the graph
         * family.
         *
         * @param m Maximum logical rows requested by the graph-family planner.
         * @param n Reserved for the generic workspace-consumer interface; the
         *          router owns its expert count in `Params`.
         * @param k Reserved for the generic workspace-consumer interface; the
         *          router owns its hidden width in `Params`.
         * @return Complete CUDA or ROCm routing workspace requirements sized
         *         for at least both `m` and the stage's concrete row count.
         */
        WorkspaceRequirements getWorkspaceRequirements(
            int m,
            int n = 0,
            int k = 0) const override;
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;
        void unbindWorkspace() override;
        bool hasWorkspace() const override;
        DeviceWorkspaceManager *getWorkspace() const override;

        /**
         * @brief Clear per-request routing metadata while preserving kernel handles.
         */
        void resetSessionState() override;
        /**
         * @brief Clear routing diagnostics while preserving captured MoE state.
         *
         * Variable row counts are owned by the request-geometry allocation and
         * are not stage-local replay metadata. This hook therefore preserves
         * only immutable launch state and backend descriptor identity.
         */
        void resetSessionStatePreservingCapturedReplay() override;
        /**
         * @brief Preserve warmed routing workspace for capture-from-Initialized.
         */
        void resetSessionStatePreservingLazyInitialization() override;
        /**
         * @brief Reset backend launch metadata owned exclusively by this stage.
         *
         * Routing stages retain independent MoE kernel instances so a sibling
         * expert or maintenance graph cannot retarget their stream, workspace,
         * or descriptor storage. A hard graph invalidation resets that private
         * backend state; captured-replay request boundaries deliberately do not
         * call this method.
         */
        void invalidateKernelDynamicState() override;

        // Test accessors
        void setMoEKernelForTesting(IMoEKernel *kernel)
        {
            params_.routed_pipeline_kernel_owner.reset();
            owned_moe_kernel_.reset();
            moe_kernel_ = kernel;
        }

        /**
         * @brief Expose the graph-lowered row ownership policy to graph tests.
         *
         * Routing publishes the runtime expert IDs consumed by the paired
         * expert stage.  A graph-construction regression must therefore prove
         * both stages received the same typed policy rather than inspecting
         * only the downstream stage and assuming the router agreed.
         */
        RoutedExpertRowExecutionPolicy
        routedExpertRowExecutionPolicyForTesting() const noexcept
        {
            return params_.routed_row_execution_policy;
        }

        /** @brief Expose the captured device row-count owner to graph tests. */
        const int32_t *activeRowCountDeviceForTesting() const noexcept
        {
            return params_.active_row_count_device;
        }

        /** @brief Expose immutable CPU routing-evidence geometry to graph tests. */
        int hostLogicalRowCountForTesting() const noexcept
        {
            return params_.host_logical_row_count;
        }

        /** @brief Expose the typed decode publication authority to graph tests. */
        MoEDecodeRoutePublicationPolicy
        decodeRoutePublicationPolicyForTesting() const noexcept
        {
            return params_.decode_route_publication;
        }

        /** @brief Expose graph-lowered device evidence collection to tests. */
        bool collectsDeviceRuntimeHistogramForTesting() const noexcept
        {
            return params_.collect_device_runtime_histogram;
        }

    private:
        Params params_;

        /// Stashed routing results for snapshot capture
        mutable std::vector<float> routing_indices_f32_;
        mutable std::vector<float> routing_weights_;
        mutable std::vector<float> router_logits_;

        /**
         * @brief Graph-stable view of the backend-owned raw router logits.
         *
         * GPU routing computes raw logits into the persistent
         * `moe_route_logits` workspace before the top-k selection kernel reads
         * them. The production route intentionally has no host mirror. This
         * non-owning tensor view lets the graph snapshot manifest record an
         * ordered device-to-device copy from that exact producer-owned address.
         * It is created while binding the graph workspace, never from execute(),
         * and is destroyed before the workspace binding is released.
         */
        std::unique_ptr<ITensor> router_logits_device_view_;

        /**
         * @brief Graph-local MoE kernel and optional non-owning test override.
         *
         * A router may share this owner only with its paired routed-expert
         * consumer so the captured producer/consumer chain can reuse the same
         * Q8 hidden publication. Separately captured main, verifier, and
         * maintenance graphs always own distinct backend objects.
         */
        mutable std::unique_ptr<IMoEKernel> owned_moe_kernel_;
        mutable IMoEKernel *moe_kernel_ = nullptr;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;

        /// Pre-allocated routing result (avoids heap allocs per decode token)
        mutable MoERoutingResult cached_routing_;
        DeviceMoELayerRuntime *moe_runtime_layer_ = nullptr;

        IMoEKernel *ensureMoEKernel() const;
        /** @brief Classify the production arithmetic route prepared for capture. */
        MoERouteLaunchKind routeLaunchKind() const noexcept;
        bool isDeviceRoutedDecodeGraphCapturable() const;
        bool isRuntimeTableDecodeGraphCapturable() const;
        bool isOverlayTicketDecodeGraphCaptureSupported() const;
        bool isOverlayTicketDecodeGraphCapturable() const;
        bool isDeviceRoutedPrefillExecutionSupported() const;
        bool isDeviceRoutedPrefillGraphCaptureSupported() const;
        bool isDeviceRoutedPrefillGraphCapturable() const;
        bool isDecodeEquivalentVerifierPrefillExecutionSupported() const;
        bool isDecodeEquivalentVerifierPrefillGraphCaptureSupported() const;
        bool isDecodeEquivalentVerifierPrefillGraphCapturable() const;
        /**
         * @brief Validate the router-owned overlay verifier ledger binding.
         *
         * @return true when the policy is disabled or the model-lifetime
         *         runtime table exposes a complete per-layer route capacity.
         */
        bool hasCompleteOverlayVerifierLedgerBinding() const;
        bool hasInitializedRuntimeTableIfProvided() const;
        bool executeDecodeEquivalentVerifierPrefill(IDeviceContext *ctx);
        void recordRuntimeHistogramTokenBoundary() const;
        /**
         * @brief Publish one CPU grouped-routing result into Dynamic evidence.
         *
         * The helper consumes only the leading `host_logical_row_count` rows
         * from `cached_routing_`.  It is deliberately unavailable to GPU
         * execution, whose evidence remains device-owned and is merged by the
         * runtime-table lifecycle at an explicit maintenance boundary.
         *
         * @param source Workload regime that produced the grouped rows.
         * @return True when no histogram is attached or the complete logical
         *         row prefix was validated and merged; false on any contract
         *         violation.
         */
        bool publishCPUGroupedRoutingEvidence(ExpertHistogramSource source) const;
        void stashRoutingResults(
            const std::vector<int> &expert_indices,
            const std::vector<float> &expert_weights,
            int seq_len, int top_k) const;

        /**
         * @brief Bind diagnostic metadata to the canonical GPU logits buffer.
         *
         * This method performs host-side setup only. It neither launches GPU
         * work nor allocates device memory. A successful binding proves that
         * snapshot capture and routing arithmetic refer to the same persistent
         * graph-workspace address.
         *
         * @return True for CPU/unbound state, or when the complete GPU logits
         *         view was bound to the current workspace.
         */
        bool bindRouterLogitsDeviceView();

    };

} // namespace llaminar2
