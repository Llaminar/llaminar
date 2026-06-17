/**
 * @file FusedGateUpGEMMStage.h
 * @brief Fused Gate/Up projection stage for FFN
 */

#pragma once

#include "../IComputeStage.h"
#include "../IWorkspaceConsumerStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"
#include "../../../loaders/WeightPlan.h"

#include <memory>
#include <optional>

namespace llaminar2
{

    // Forward declarations
    class ITensorFusedGateUpGemm;
    class FP32Tensor;
    class PreparedWeightStore;
    class TensorBase;

    /**
     * @brief Fused Gate/Up projection stage for FFN
     *
     * Efficiently computes both gate and up projections from a shared input
     * by quantizing the input once and reusing the Q8_1 buffer for both
     * projections. This avoids redundant quantization and improves cache locality.
     *
     * Pattern: input → [gate_proj, up_proj] → SwiGLU → down_proj
     *
     * This stage delegates to CPUQuantisedGemmKernel::multiply_fused_multi(), which
     * handles the quantization and multi-projection execution internally.
     *
     * **Workspace Management (Phase 4)**:
     * Implements IWorkspaceConsumerStage to delegate workspace requirements to the
     * underlying fused GEMM kernel. This enables zero-allocation GPU execution by
     * pre-binding workspace buffers during graph setup.
     */
    class FusedGateUpGEMMStage : public IComputeStage, public IWorkspaceConsumerStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            // Type-safe tensor pointers (required)
            const ITensor *input = nullptr; ///< Input activation tensor [m, k]
            int m = 0;                      ///< Batch size * seq_len
            int k = 0;                      ///< Input dimension (d_model)

            // Gate projection
            const ITensor *w_gate = nullptr;
            ITensor *output_gate = nullptr;
            int n_gate = 0;
            const TensorBase *bias_gate = nullptr;

            // Up projection
            const ITensor *w_up = nullptr;
            ITensor *output_up = nullptr;
            int n_up = 0;
            const TensorBase *bias_up = nullptr;

            // Optional BufferIds for contract-based coherence
            std::optional<BufferId> input_buffer_id;
            std::optional<BufferId> output_gate_buffer_id;
            std::optional<BufferId> output_up_buffer_id;

            // =================================================================
            // Phase 7: PreparedWeightRef for direct kernel resolution
            // =================================================================
            std::optional<PreparedWeightRef> prepared_ref_gate;
            std::optional<PreparedWeightRef> prepared_ref_up;
            PreparedWeightStore *prepared_store = nullptr;

            /**
             * @brief Execute tiny verifier batches as repeated M=1 gate/up GEMVs.
             *
             * The vLLM-style MTP verifier may batch M=2..4 rows into one graph,
             * but accepted-state publication must be decode-equivalent.  This
             * flag keeps the public output tensors shaped as [M, N] while each
             * row uses the same fused gate/up route as ordinary one-token decode.
             */
            bool force_decode_equivalent_verifier_prefill = false;
        };

        explicit FusedGateUpGEMMStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        bool validatePreparedWeights(std::string *error) const override;
        ComputeStageType type() const override { return ComputeStageType::GEMM_FUSED_GATE_UP; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;

        // =============================================================================
        // IWorkspaceConsumerStage Implementation
        // =============================================================================
        IWorkspaceConsumer *getKernelAsWorkspaceConsumer() override;
        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;

    private:
        Params params_;
        ITensorFusedGateUpGemm *cached_kernel_ = nullptr; ///< Cached for workspace binding

        ITensorFusedGateUpGemm *resolvePreparedKernel(const char *caller);
        bool executeDecodeEquivalentVerifierPrefill(
            IDeviceContext *ctx,
            const TensorBase *input,
            TensorBase *output_gate,
            TensorBase *output_up,
            ITensorFusedGateUpGemm *kernel);

        std::shared_ptr<FP32Tensor> verifier_input_row_;
        std::shared_ptr<FP32Tensor> verifier_gate_row_;
        std::shared_ptr<FP32Tensor> verifier_up_row_;
    };

} // namespace llaminar2
