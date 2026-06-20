/**
 * @file GEMMStage.h
 * @brief GEMM stage: C = alpha * A * B + beta * C
 */

#pragma once

#include "../IComputeStage.h"
#include "../IWorkspaceConsumerStage.h"
#include "../StageParamsBase.h"
#include "../../../utils/GemmContext.h"
#include "../../../memory/BufferId.h"
#include "../../../loaders/WeightPlan.h"

#include <memory>
#include <optional>

namespace llaminar2
{

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
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            // Type-safe tensor pointers (required)
            const ITensor *A = nullptr; ///< Input activation tensor [m, k]
            const ITensor *B = nullptr; ///< Weight tensor [k, n] (may be quantized)
            ITensor *C = nullptr;       ///< Output tensor [m, n]
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
             * @brief Bias tensor (preferred interface)
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
             * use a grouped M=2..4 path with the serial-decode numerical contract.
             */
            bool force_decode_equivalent_verifier_prefill = false;

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

    private:
        Params params_;

        // === Cached kernel pointers (avoid KernelFactory mutex per execute) ===
        // These are populated on first execute() and reused thereafter.
        // The KernelFactory owns the lifetime of these objects (they're in
        // registry maps), so raw pointers are safe here.
        ITensorGemm *cached_gemm_ = nullptr;
        bool cache_resolved_ = false;

        bool executeDecodeEquivalentVerifierPrefill(
            const TensorBase *A_base,
            TensorBase *C_base,
            ITensorGemm *gemm,
            int effective_n);

        // Reused one-row verifier scratch. Keeping this state on the stage avoids
        // allocating transient tensors while the hot verifier path is iterating rows.
        std::shared_ptr<FP32Tensor> verifier_gate_row_;
        std::shared_ptr<FP32Tensor> verifier_input_row_;
        std::shared_ptr<FP32Tensor> verifier_output_row_;
    };

} // namespace llaminar2
