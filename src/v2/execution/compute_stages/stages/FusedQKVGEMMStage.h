/**
 * @file FusedQKVGEMMStage.h
 * @brief Fused Q/K/V projection stage
 */

#pragma once

#include "../IComputeStage.h"
#include "backends/DeviceId.h"

namespace llaminar2
{

    /**
     * @brief Fused Q/K/V projection stage
     *
     * Efficiently computes multiple linear projections (Q, K, V) from a shared
     * input by quantizing the input once and reusing the Q8_1 buffer for all
     * projections. This avoids redundant quantization and improves cache locality.
     *
     * This stage delegates to QuantisedGemmKernel::multiply_fused_multi(), which
     * handles the quantization and multi-projection execution internally.
     */
    class FusedQKVGEMMStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Type-safe tensor pointers (required)
            const ITensor *input = nullptr; ///< Input activation tensor [m, k]
            int m = 0;                      ///< Batch size * seq_len
            int k = 0;                      ///< Input dimension (d_model)

            // Q projection
            const ITensor *wq = nullptr;
            ITensor *output_q = nullptr;
            int n_q = 0;
            const float *bias_q = nullptr;

            // K projection
            const ITensor *wk = nullptr;
            ITensor *output_k = nullptr;
            int n_k = 0;
            const float *bias_k = nullptr;

            // V projection
            const ITensor *wv = nullptr;
            ITensor *output_v = nullptr;
            int n_v = 0;
            const float *bias_v = nullptr;

            // Device target for kernel dispatch
            DeviceId device_id = DeviceId::cpu();
        };

        explicit FusedQKVGEMMStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::GEMM_FUSED_QKV; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;
    };

} // namespace llaminar2
