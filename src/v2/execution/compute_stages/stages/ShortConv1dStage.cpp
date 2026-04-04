/**
 * @file ShortConv1dStage.cpp
 * @brief Implementation of causal depthwise conv1d + SiLU for GDN layers
 *
 * Delegates to ITensorShortConvolution kernel for the actual computation.
 * Stage handles tensor extraction, null checks, and buffer contract management.
 */

#include "ShortConv1dStage.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../utils/Logger.h"

namespace llaminar2
{

    ShortConv1dStage::ShortConv1dStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool ShortConv1dStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "ShortConv1dStage"))
            return false;

        if (!ensureRequiredPointers("ShortConv1dStage",
                                    {{"input", params_.input},
                                     {"output", params_.output},
                                     {"weight", params_.weight}}))
            return false;

        if (!params_.kernel)
        {
            LOG_ERROR("[ShortConv1dStage] kernel (ITensorShortConvolution) not set");
            return false;
        }

        auto *input_base = requireTensorBasePtr(params_.input, "input");
        auto *output_base = requireTensorBasePtr(params_.output, "output");
        auto *weight_base = requireTensorBasePtr(params_.weight, "weight");

        if (!input_base || !output_base || !weight_base)
            return false;

        const float *input_data = input_base->data();
        float *output_data = output_base->mutable_data();
        const float *weight_data = weight_base->data();

        const float *bias_data = nullptr;
        if (params_.bias)
        {
            auto *bias_base = dynamic_cast<TensorBase *>(const_cast<ITensor *>(params_.bias));
            if (bias_base)
                bias_data = bias_base->data();
        }

        if (!input_data || !output_data || !weight_data)
        {
            LOG_ERROR("[ShortConv1dStage] Null data pointer");
            return false;
        }

        bool ok = params_.kernel->forward(
            input_data, weight_data, bias_data,
            output_data, params_.conv_state,
            params_.seq_len, params_.channels, params_.kernel_size,
            /*apply_silu=*/true);

        if (!ok)
        {
            LOG_ERROR("[ShortConv1dStage] Kernel forward() failed");
            return false;
        }

        LOG_DEBUG("[ShortConv1dStage] Executed: seq_len=" << params_.seq_len
                                                          << " channels=" << params_.channels
                                                          << " kernel=" << params_.kernel_size
                                                          << (params_.seq_len == 1 ? " (decode)" : " (prefill)"));
        return true;
    }

    size_t ShortConv1dStage::estimatedFlops() const
    {
        const size_t S = static_cast<size_t>(params_.seq_len);
        const size_t C = static_cast<size_t>(params_.channels);
        const size_t K = static_cast<size_t>(params_.kernel_size);
        // Per output: K MAC + SiLU (4 ops)
        return S * C * (2 * K + 4);
    }

    size_t ShortConv1dStage::estimatedMemoryBytes() const
    {
        const size_t S = static_cast<size_t>(params_.seq_len);
        const size_t C = static_cast<size_t>(params_.channels);
        const size_t K = static_cast<size_t>(params_.kernel_size);
        return (S * C + C * K + S * C) * sizeof(float); // input + weight + output
    }

    bool ShortConv1dStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::CPU;
    }

    StageDumpInfo ShortConv1dStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        // Use addOutput with proper rows/cols so snapshot capture works
        auto *out_base = dynamic_cast<const TensorBase *>(params_.output);
        if (out_base)
            info.addOutput("output", params_.output, out_base->rows(), out_base->cols());

        return info;
    }

    StageBufferRequirements ShortConv1dStage::getBufferRequirements() const
    {
        return {};
    }

    StageBufferContract ShortConv1dStage::bufferContract() const
    {
        StageBufferContract contract;
        if (params_.input_buffer_id)
            contract.addInput(*params_.input_buffer_id);
        if (params_.output_buffer_id)
            contract.addOutput(*params_.output_buffer_id);
        return contract;
    }

} // namespace llaminar2
