/** @file GEMMStage.h
 * @brief GEMM stage: C = alpha * A * B + beta * C
 * Verifier scopes borrow device counts; adapters retain physical scratch and exact stream ordering.
 * Original causal query work is a separate retained descriptor; it never selects
 * verifier arithmetic or invents a second live-count owner.
 * Typed bias retains its original tensor/view through ordinary and verifier dispatch.
 * HC two-term input publication retains the original activated FP32 value,
 * backend nonlinear arithmetic and prepared projection/workspace authority.
 */

#pragma once

#include <array>
#include <vector>
#include "tensors/TensorType.h"
#include "execution/local_execution/device/WorkspaceBufferLease.h"
#include "utils/PerfStatsCollector.h"

#include "kernels/common/DeviceRowRange.h"
#include "kernels/common/DeviceRequestRowRanges.h"
#include "kernels/common/Q8ResidualCorrection.h"
#include "kernels/common/GatedRMSNormProjection.h"

#include "../IComputeStage.h"
#include "../ProjectionRowWorkBinding.h"
#include "../IWorkspaceConsumerStage.h"
#include "../StageParamsBase.h"
#include "../../../utils/GemmContext.h"
#include "../../../memory/BufferId.h"
#include "../../../loaders/WeightPlan.h"

#include <memory>
#include <optional>

namespace llaminar2
{
    class CapturedDeviceStorageLease;

    // Forward declarations
    class ITensorGemm;
    class FP32Tensor;
    class PreparedWeightStore;

    /**
     * @brief GEMM stage: C = alpha * A * B + beta * C
     *
     * Takes FP32 activations and quantized/FP32 weights, handles quantization internally.
     * For multi-projection patterns (Q/K/V or gate/up), use FusedQKVGEMMStage or
     * FusedGateUpGEMMStage which efficiently share quantization across projections.
     *
     * **Tensor Parallelism Support (Phase 2)**:
     * When `output_range` is set, executes a row-sliced GEMM for tensor parallelism:
      * - Uses `PreparedWeightStore::slicedGemmKernel()` to resolve a prepared slice
     * - Only computes output rows in [output_range.start, output_range.end)
     * - Caller is responsible for MPI AllReduce after execution if needed
     *
     * **Workspace Management (Phase 4)**:
     * Implements IWorkspaceConsumerStage to delegate workspace requirements to the
     * underlying GEMM kernel. This enables zero-allocation GPU execution by pre-binding
     * workspace buffers during graph setup.
     */
    class GEMMStage : public IComputeStage, public IWorkspaceConsumerStage
    {
    public:
        /** @brief Retained FP32 result of HC /4 SiLU, produced before two-term projection. */
        struct HCResidualDownPublication {
            TensorBase* activated; ///< Existing arena result, never a new allocation.
            BufferId buffer; ///< Exact output coherence authority.
        };
        /** @brief Immutable gate/gain input for one retained FP32 normalization+projection. */
        struct GatedNormPublication {
            const ITensor* gate=nullptr; ///< Existing read-only arena gate output.
            const ITensor* gamma=nullptr; ///< Original model gain tensor.
            BufferId gate_buffer; ///< Exact gate coherence authority.
            GatedRMSNormProjectionParameters parameters; ///< Original norm arithmetic.
        };
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            // Type-safe tensor pointers (required)
            const ITensor *A = nullptr; ///< Input activation tensor [m, k]
            const ITensor *B = nullptr; ///< Weight tensor [k, n] (may be quantized)
            ITensor *C = nullptr;       ///< Output tensor [m, n]
            std::optional<HCResidualDownPublication> hc_residual_down; ///< Typed raw-down input transform; valid only for TwoTermQ8.
            std::optional<GatedNormPublication> gated_norm; ///< Explicit ordinary scalar SingleQ8 input program.
            int m = 0, n = 0, k = 0;    ///< Matrix dimensions
            float alpha = 1.0f;         ///< Scale factor for A*B
            float beta = 0.0f;          ///< Scale factor for existing C
            bool transpose_B = false;   ///< Whether B is transposed (n × k)

            // =================================================================
            // Bias Configuration (Enhanced for Tensor Parallelism)
            // =================================================================

            /**
             * @brief Raw bias pointer (legacy interface)
             * @deprecated Use bias_tensor instead for TensorSlice/TP compatibility.
             */
            const float *bias = nullptr;

            /**
             * @brief Original FP32 bias tensor/view, retained through ordinary and verifier execution.
             */
            const ITensor *bias_tensor = nullptr;

            /**
             * @brief Whether bias is required for this operation
             */
            bool bias_required = false;

            /**
             * @brief Get bias data pointer from either source
             */
            const float *getBiasData() const;

            /**
             * @brief Validate params and throw on configuration error
             */
            void validate(const std::string &stage_name = "GEMMStage") const;

            // SwiGLU fusion (extended fields)
            const ITensor *gate_input = nullptr;
            bool do_swiglu = false;

            // device_id and mpi_ctx inherited from StageParamsBase

            // =================================================================
            // Tensor Parallelism Parameters (Phase 2)
            // =================================================================

            /**
             * @brief Output row range for row-parallel GEMM
             */
            WorkRange output_range;

            /**
             * @brief Whether this GEMM requires AllReduce after execution
             */
            bool needs_allreduce = false;

            /**
             * @brief GEMM profiling context (attention, FFN, LM head)
             *
             * Used by GPU kernel profilers to attribute GEMM time to the
             * correct functional category. Set by the graph builder.
             */
            GemmContext gemm_context = GemmContext::NONE;

            // Optional BufferIds for contract-based coherence
            std::optional<BufferId> a_buffer_id;
            std::optional<BufferId> gate_buffer_id;
            std::optional<BufferId> c_buffer_id;

            /**
             * @brief Execute tiny verifier batches with decode-equivalent grouped GEMV.
             *
             * MTP all-position publication restores state from individual verifier
             * rows. Quantized kernels may choose different M=2..4 dispatches than
             * serial decode's M=1 route, and even tiny output-projection drift can
             * flip later MoE routes. When enabled, this stage preserves the same
             * graph-level output tensor but requires the kernel layer to prove and
             * use an economical grouped path at every positive M with the
             * serial-decode numerical contract.
             */
            bool force_decode_equivalent_verifier_prefill = false;
            /// Immutable verifier geometry; its borrowed count is ordered by the graph producer.
            std::optional<DeviceRowRange> verifier_row_range;
            /// Independent floating request prefixes; requires explicit decode-equivalent arithmetic and excludes verifier_row_range.
            std::optional<DeviceRequestRowRanges> request_rows;
            /// Explicit decode-only causal work; plain projections keep their original arithmetic.
            std::optional<ProjectionRowWorkDeclaration> projection_row_work;

            // =================================================================
            // Phase 7: PreparedWeightRef for direct kernel resolution
            // =================================================================

            /**
             * @brief Prepared weight reference for this GEMM's weight tensor.
             * When set (together with prepared_store), the stage resolves its
             * kernel via PreparedWeightStore::gemmKernel() or slicedGemmKernel().
             */
            std::optional<PreparedWeightRef> prepared_ref;

            /**
             * @brief PreparedWeightStore for resolving prepared_ref.
             * Lifetime managed by DeviceGraphOrchestrator.
             */
            PreparedWeightStore *prepared_store = nullptr;
            /// Explicit immutable operand precision; existing model defaults retain SingleQ8.
            QuantizedActivationOperandPrecision quantized_operand_precision=QuantizedActivationOperandPrecision::SingleQ8;
        };

        explicit GEMMStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        bool validatePreparedWeights(std::string *error) const override;
        ComputeStageType type() const override { return ComputeStageType::GEMM; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;
        bool requiresAllreduce() const override { return params_.needs_allreduce; }
        /** @brief Include the explicit arithmetic/quantizer and static row geometry in capture identity.
         * Uncounted SingleQ8 stages keep their established zero variant. Explicit
         * prefix/request selections and TwoTermQ8 hash complete immutable row
         * metadata and arithmetic policy; count values are never host-observed.
         * HC compares its frozen native binding as a validation side effect;
         * addresses/leases never change the signature during materialization.
         * Existing prepared/arena ownership remains the allocation authority.
         */
        uint64_t graphCaptureVariantSignature() const override;
        /** @brief Freeze HC native operands after the executor's allocation-only prebind.
         * @param ctx Exact participant context; no GPU query or work is issued.
         * @param stream Exact non-default recording stream.
         * @return True after first binding or identical preparatory reentry.
         * @details Retains only the four original TwoTerm workspace allocations
         * through existing PMA-backed leases; it does not allocate GPU storage.
         * @throws std::logic_error On any postbind operand/workspace replacement.
         */
        bool prepareGraphLaunch(IDeviceContext* ctx,void* stream) override;
        /** @brief GPU metadata binds before capture; CPU query metadata binds before input observation. */
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override {
            if(params_.projection_row_work&&params_.device_id.is_cpu())return GraphLaunchPreparationPolicy::HostEntry;
            return (params_.projection_row_work||params_.hc_residual_down||params_.gated_norm)&&params_.device_id.is_gpu()
                ? GraphLaunchPreparationPolicy::CaptureOnly : GraphLaunchPreparationPolicy::None;
        }
        void resetSessionState() override;
        void resetSessionStatePreservingCapturedReplay() override;
        void resetSessionStatePreservingLazyInitialization() override;

        /// Target device for coherence management

        const Params &getParams() const { return params_; }

        // =================================================================
        // IWorkspaceConsumerStage Implementation
        // =================================================================

        /**
         * @brief Get the GEMM kernel as IWorkspaceConsumer for delegation
         *
         * Fetches the prepared kernel from PreparedWeightStore. The same kernel
         * is returned on every call for this stage.
         *
         * @return Kernel implementing IWorkspaceConsumer, or nullptr if not available
         */
        IWorkspaceConsumer *getKernelAsWorkspaceConsumer() override;
        /** @brief Bind the admitted arena and prepare CPU address metadata before validation. */
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;

        /** @brief Declare projection scratch and the explicitly selected input transform. */
        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;

    private:
        Params params_;
        std::unique_ptr<ProjectionRowWorkBinding> projection_row_work_; ///< Cold original-owner binding, retained across request reset.
        /** @brief Exact address/geometry identity; contains no live GPU values. */
        struct HCNativePublicationIdentity {
            std::array<const void*,7> pointers{}; ///< Raw A, activated, C, then the four consumed banks.
            std::array<size_t,7> bytes{}; ///< Complete retained tensor/bank extents.
            std::array<TensorType,3> types{}; ///< Original tensor storage representations.
            std::array<const void*,4> weight_planes{}; ///< Prepared payload/scales/mins/emins.
            std::array<uint64_t,12> weight_geometry{}; ///< Every nonpointer native descriptor field.
            uint64_t workspace_id=0; ///< Original manager instance, never a new epoch counter.
            ITensorGemm* engine=nullptr; ///< Original prepared engine resolved from the store.
            friend bool operator==(const HCNativePublicationIdentity&,const HCNativePublicationIdentity&)=default;
        };
        /** @brief Immutable captured ownership, published once after allocation prebind. */
        struct HCNativePublicationBinding {
            HCNativePublicationIdentity identity;
            std::array<std::vector<size_t>,3> shapes; ///< Exact complete tensor shapes, copied only at setup.
            std::array<std::shared_ptr<const CapturedDeviceStorageLease>,3> operands; ///< Original mutable activation allocation borrows.
            std::array<std::shared_ptr<const WorkspaceBufferLease>,4> banks; ///< Original PMA-owned allocation leases.
            void* stream=nullptr; ///< Exact recording stream; preparatory reentry cannot replace it.
        };
        std::optional<HCNativePublicationBinding> hc_native_binding_; ///< Survives data reset; only stage destruction retires it.
        /** @brief Inspect immutable addresses/descriptors without GPU work or host value reads.
         * @return Complete consumed native identity; no ownership is created here.
         * @throws std::logic_error For incomplete, foreign or unprepared operands/banks.
         */
        HCNativePublicationIdentity currentHCNativePublicationIdentity() const;
        /** @brief Reject unbound/replaced native identity, shape or bank ownership before capture/replay.
         * @throws std::logic_error For any stale pointer, extent, source, shape, stream or generation.
         */
        void requireHCNativePublicationBinding() const;

        /** @brief Exact prepared scalar producer/consumer address identity, never live data. */
        struct GatedNormIdentity {
            std::array<const void*,9> pointers{}; ///< A/gate/gamma/C, then original five workspace banks.
            std::array<size_t,9> bytes{}; ///< Complete original retained capacities.
            std::array<TensorType,4> types{}; ///< Actual tensor storage representations.
            std::array<const void*,4> weight_planes{}; ///< Original native immutable planes.
            std::array<uint64_t,12> weight_geometry{}; ///< Complete descriptor fields.
            uint64_t workspace_id=0; ///< Original PMA-backed workspace instance.
            ITensorGemm* engine=nullptr; ///< Exact prepared-store engine.
            friend bool operator==(const GatedNormIdentity&,const GatedNormIdentity&)=default;
        };
        /** @brief Capture-held borrows of existing tensor and workspace authorities. */
        struct GatedNormBinding {
            GatedNormIdentity identity;
            std::array<std::vector<size_t>,4> shapes;
            std::array<std::shared_ptr<const CapturedDeviceStorageLease>,4> operands;
            std::array<std::shared_ptr<const WorkspaceBufferLease>,5> banks;
            void* stream=nullptr; ///< Exact recording stream, immutable after preparation.
        };
        std::optional<GatedNormBinding> gated_norm_binding_; ///< Survives data-only request resets.
        /** @return Original allocation-prebound tensors, engine, native source and scratch identity.
         * @throws std::logic_error For missing/foreign owner, source or native storage.
         */
        GatedNormIdentity currentGatedNormIdentity() const;
        /** @brief Reject any post-preparation pointer/capacity/type/shape/source/stream replacement. */
        void requireGatedNormBinding() const;
        /** @brief Freeze the exact retained composite owners at the existing cold capture boundary. */
        bool prepareGatedNormBinding(IDeviceContext* ctx,void* stream);

        // === Cached kernel pointers (avoid KernelFactory mutex per execute) ===
        // These are populated on first execute() and reused thereafter.
        // The KernelFactory owns the lifetime of these objects (they're in
        // registry maps), so raw pointers are safe here.
        ITensorGemm *cached_gemm_ = nullptr;
        bool cache_resolved_ = false;

        /** @brief Actual selected API, independent of a caller's verifier request flag. */
        enum class HCUpSubmissionOperation { Ordinary, DecodeEquivalent, FusedSwiGLU, ResidualCorrected };
        /** @brief Bind original HC metadata once, independently of collection.
         * @throws std::invalid_argument For foreign immutable source or policy.
         * @details No device count is read. Keys own no mutable counter state.
         */
        void initializeHCUpSubmissionIdentities();
        /** @brief Count one successful host submission with setup-bound metadata.
         * @param columns Actual output width, authenticated against setup geometry.
         * @param operation Actual API that succeeded, never inferred from request flags.
         * @details CPU team zero observes once; GPU replay does not call execute.
         */
        void recordHCUpSubmission(int columns, HCUpSubmissionOperation operation);
        std::array<std::optional<PerfStatsCollector::CounterIdentity>, 4> hc_submission_identities_;

        bool executeDecodeEquivalentVerifierPrefill(
            const TensorBase *A_base,
            TensorBase *C_base,
            ITensorGemm *gemm,
            int effective_n);
        void clearCachedGemmStream();

    };

} // namespace llaminar2
