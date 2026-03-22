/**
 * @file FusedQKVGEMMStage.h
 * @brief Fused Q/K/V projection stage
 */

#pragma once

#include "../IComputeStage.h"
#include "../IWorkspaceConsumerStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"

#include <optional>

namespace llaminar2
{

    // Forward declarations for cached kernel pointers
    class ITensorGemm;

    /**
     * @brief Fused Q/K/V projection stage
     *
     * Efficiently computes multiple linear projections (Q, K, V) from a shared
     * input. Uses individual ITensorGemm kernels with multiply_fused() for shared
     * input quantization, aligned with the FusedGateUpGEMM pattern.
     *
     * Implements IWorkspaceConsumerStage to delegate workspace requirements to the
     * underlying GEMM kernels for GPU execution.
     */
    class FusedQKVGEMMStage : public IComputeStage, public IWorkspaceConsumerStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            // Type-safe tensor pointers (required)
            const ITensor *input = nullptr; ///< Input activation tensor [m, k]
            int m = 0;                      ///< Batch size * seq_len
            int k = 0;                      ///< Input dimension (d_model)

            // Q projection
            const ITensor *wq = nullptr;
            ITensor *output_q = nullptr;
            int n_q = 0;
            const TensorBase *bias_q = nullptr; ///< Optional bias tensor for tensor-aware GPU path

            // K projection
            const ITensor *wk = nullptr;
            ITensor *output_k = nullptr;
            int n_k = 0;
            const TensorBase *bias_k = nullptr; ///< Optional bias tensor for tensor-aware GPU path

            // V projection
            const ITensor *wv = nullptr;
            ITensor *output_v = nullptr;
            int n_v = 0;
            const TensorBase *bias_v = nullptr; ///< Optional bias tensor for tensor-aware GPU path

            // Optional BufferIds for contract-based coherence
            std::optional<BufferId> input_buffer_id;
            std::optional<BufferId> output_q_buffer_id;
            std::optional<BufferId> output_k_buffer_id;
            std::optional<BufferId> output_v_buffer_id;
        };

        explicit FusedQKVGEMMStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::GEMM_FUSED_QKV; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;

        // =================================================================
        // IWorkspaceConsumerStage Implementation
        // =================================================================

        /**
         * @brief Get a GEMM kernel as IWorkspaceConsumer for delegation
         *
         * Returns the Q projection kernel from KernelFactory. All three kernels
         * (Q, K, V) share the same workspace, so returning any one works for
         * workspace requirements calculation.
         *
         * @return Kernel implementing IWorkspaceConsumer, or nullptr if not available
         */
        IWorkspaceConsumer *getKernelAsWorkspaceConsumer() override;

        /**
         * @brief Bind workspace to ALL THREE underlying GEMM kernels (Q, K, V)
         *
         * Override the default single-kernel binding to bind all three projection
         * kernels since they each need workspace for GPU execution.
         */
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;

        /**
         * @brief Unbind workspace from all three kernels
         */
        void unbindWorkspace() override;

    private:
        Params params_;

        // === Cached kernel pointers (avoid KernelFactory mutex per execute) ===
        ITensorGemm *cached_gemm_q_ = nullptr;
        ITensorGemm *cached_gemm_k_ = nullptr;
        ITensorGemm *cached_gemm_v_ = nullptr;
        bool cache_resolved_individual_ = false;
    };

} // namespace llaminar2
