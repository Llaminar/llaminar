/**
 * @file RMSNormStage.h
 * @brief RMS normalization stage
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include <memory>

namespace llaminar2
{
    // Forward declaration
    class ITensorRMSNorm;

    /**
     * @brief RMS normalization stage
     *
     * Type-safe implementation using TensorBase* instead of void*.
     * The tensor's native_type() determines precision dispatch.
     * Uses IActivationTensor::applyRMSNorm() for polymorphic device dispatch.
     *
     * Device-agnostic: IActivationTensor handles CPU/GPU dispatch internally.
     */
    class RMSNormStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            // Type-safe tensor pointers (required)
            ITensor *input = nullptr;       ///< Input activation tensor (IActivationTensor*)
            ITensor *output = nullptr;      ///< Output tensor (can be same as input for in-place)
            const ITensor *gamma = nullptr; ///< Gamma weights tensor

            float eps = 1e-6f; ///< Epsilon for numerical stability

            // Explicit seq_len override for pre-allocated buffers
            // If 0, derives from input tensor dimensions
            // CRITICAL: Must be set during decode when using pre-allocated buffers
            int seq_len = 0;
        };

        explicit RMSNormStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::RMS_NORM; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;

        /// Target device for coherence management


        const Params &getParams() const { return params_; }

    private:
        Params params_;
        mutable std::unique_ptr<llaminar2::ITensorRMSNorm> cached_kernel_;
    };

} // namespace llaminar2
