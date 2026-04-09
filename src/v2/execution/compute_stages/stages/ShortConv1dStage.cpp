/**
 * @file ShortConv1dStage.cpp
 * @brief Implementation of causal depthwise conv1d + SiLU for GDN layers
 *
 * Delegates to ITensorShortConvolution kernel for the actual computation.
 * Stage handles tensor extraction, null checks, and buffer contract management.
 *
 * GPU path: Uses ensureOnDevice() / allocateOnDevice() / gpu_data_ptr() to
 * keep data on-device. No H2D/D2H copies in the hot path.
 *
 * CPU path: Uses data() / mutable_data() host pointers.
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

        // Bind stage stream to kernel before execution
        params_.kernel->setGPUStream(gpuStream());

        auto *input_base = requireTensorBasePtr(params_.input, "input");
        auto *output_base = requireTensorBasePtr(params_.output, "output");
        auto *weight_base = requireTensorBasePtr(params_.weight, "weight");

        if (!input_base || !output_base || !weight_base)
            return false;

        // =================================================================
        // GPU path: keep data on-device, pass device pointers to kernel
        // =================================================================
        if (device().is_gpu())
        {
            // Ensure inputs on device (const_cast is safe: ensureOnDevice
            // is logically const — changes placement, not tensor values)
            const_cast<TensorBase *>(input_base)->ensureOnDevice(device());
            const_cast<TensorBase *>(weight_base)->ensureOnDevice(device());

            // For in-place operation (input == output), allocateOnDevice is skipped
            // since input is already on device. For separate output, allocate.
            if (params_.input != params_.output)
                const_cast<TensorBase *>(output_base)->allocateOnDevice(device());

            const float *d_input = static_cast<const float *>(input_base->gpu_data_ptr());
            const float *d_weight = static_cast<const float *>(weight_base->gpu_data_ptr());
            float *d_output = static_cast<float *>(const_cast<TensorBase *>(output_base)->gpu_data_ptr());

            const float *d_bias = nullptr;
            if (params_.bias)
            {
                auto *bias_base = dynamic_cast<const TensorBase *>(params_.bias);
                if (bias_base)
                {
                    const_cast<TensorBase *>(bias_base)->ensureOnDevice(device());
                    d_bias = static_cast<const float *>(bias_base->gpu_data_ptr());
                }
            }

            bool ok = params_.kernel->forward(
                d_input, d_weight, d_bias,
                d_output, params_.conv_state,
                params_.seq_len, params_.channels, params_.kernel_size,
                /*apply_silu=*/true);

            if (!ok)
            {
                LOG_ERROR("[ShortConv1dStage] GPU kernel forward() failed");
                return false;
            }

            // Mark output as device-authoritative (kernel wrote to GPU buffer)
            const_cast<TensorBase *>(output_base)->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device());

            LOG_DEBUG("[ShortConv1dStage] GPU: seq_len=" << params_.seq_len
                                                         << " channels=" << params_.channels
                                                         << " kernel=" << params_.kernel_size
                                                         << (params_.seq_len == 1 ? " (decode)" : " (prefill)"));
            return true;
        }

        // =================================================================
        // CPU path: use host pointers
        // =================================================================
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
        switch (backend)
        {
        case ComputeBackendType::CPU:
            return true;
#ifdef HAVE_CUDA
        case ComputeBackendType::GPU_CUDA:
            return true;
#endif
#ifdef HAVE_ROCM
        case ComputeBackendType::GPU_ROCM:
            return true;
#endif
        default:
            return false;
        }
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
