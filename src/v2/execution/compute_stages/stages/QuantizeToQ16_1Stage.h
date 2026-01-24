/**
 * @file QuantizeToQ16_1Stage.h
 * @brief Quantize FP32 tensor to Q16_1 format
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"

namespace llaminar2
{

    /**
     * @brief Quantize FP32 tensor to Q16_1 format
     *
     * Used to initialize the Q16_1 residual stream from FP32 embedding output
     * in HybridQ16 mode. This stage copies and quantizes FP32 → Q16_1.
     *
     * Typically used once at the start of inference to initialize the
     * typed residual stream before layer 0 runs.
     */
    class QuantizeToQ16_1Stage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            const ITensor *input = nullptr; ///< FP32 input [num_elements]
            ITensor *output = nullptr;      ///< Q16_1 output [num_elements]
            size_t num_elements = 0;           ///< Number of elements to quantize
        };

        explicit QuantizeToQ16_1Stage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::QUANTIZE; }
        size_t estimatedFlops() const override { return params_.num_elements * 4; }
        size_t estimatedMemoryBytes() const override { return params_.num_elements * 6; }
        bool supportsBackend(ComputeBackendType backend) const override { return backend == ComputeBackendType::CPU; }
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;
    };

} // namespace llaminar2
