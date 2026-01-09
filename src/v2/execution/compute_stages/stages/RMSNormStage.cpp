/**
 * @file RMSNormStage.cpp
 * @brief Implementation of RMSNormStage
 */

#include "RMSNormStage.h"
#include "../ComputeStageUtils.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../kernels/KernelFactory.h"

namespace llaminar2
{

    // =============================================================================
    // RMSNormStage Implementation (Type-Safe via IActivationTensor)
    // =============================================================================

    RMSNormStage::RMSNormStage(Params params) : params_(std::move(params)) {}

    bool RMSNormStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[RMSNormStage] Null device context");
            return false;
        }

        if (!params_.input || !params_.output || !params_.gamma)
        {
            LOG_ERROR("[RMSNormStage] Null tensor(s): input=" << params_.input
                                                              << " output=" << params_.output
                                                              << " gamma=" << params_.gamma);
            return false;
        }

        // Cast ITensor* to TensorBase* for CPU operations
        auto *input_base = requireTensorBase(params_.input, "input");
        auto *gamma_base = requireTensorBase(params_.gamma, "gamma");
        auto *output_base = requireTensorBase(params_.output, "output");
        if (!input_base || !gamma_base || !output_base)
        {
            LOG_ERROR("[RMSNormStage] GPU tensors not yet supported");
            return false;
        }

        // === Stage Tracing (Task 3) ===
        traceInput("input", params_.input);
        traceInput("gamma", params_.gamma);

        // Use explicit seq_len if provided, otherwise derive from tensor dimensions
        // CRITICAL: For pre-allocated buffers, params_.seq_len must be set to avoid
        // processing garbage data beyond the actual sequence
        const int seq_len = params_.seq_len > 0 ? params_.seq_len : static_cast<int>(input_base->rows());
        const int hidden_dim = static_cast<int>(input_base->cols());

        // Create kernel via KernelFactory with automatic type dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(params_.device_id);

        LOG_DEBUG("[RMSNormStage] Execute: seq_len=" << seq_len
                                                     << " hidden_dim=" << hidden_dim
                                                     << " eps=" << params_.eps
                                                     << " tensor_type=" << input_base->dtype_name()
                                                     << " device=" << static_cast<int>(dev_type));

        // DEBUG: Log input for parity debugging
        const float *in_data = params_.input->fp32_data();
        if (in_data)
        {
            LOG_DEBUG("[RMSNormStage] input[0:8]=" << std::setprecision(6)
                                                   << in_data[0] << "," << in_data[1] << "," << in_data[2] << "," << in_data[3] << ","
                                                   << in_data[4] << "," << in_data[5] << "," << in_data[6] << "," << in_data[7]
                                                   << " device_id=" << params_.device_id.to_string());
        }
        auto kernel = llaminar::v2::kernels::KernelFactory::createRMSNorm(input_base, dev_type);
        if (!kernel)
        {
            LOG_ERROR("[RMSNormStage] Failed to create RMSNorm kernel for type "
                      << input_base->dtype_name());
            return false;
        }

        // apply_tensor handles input != output cases internally
        bool success = kernel->apply_tensor(
            input_base,
            gamma_base,
            output_base,
            seq_len,
            hidden_dim,
            params_.eps,
            params_.mpi_ctx,
            params_.device_id.toKernelDeviceIndex());

        // DEBUG: Log RMSNorm output for parity debugging
        if (success)
        {
            // Get output from host after GPU kernel (if GPU, needs sync)
            const float *out_data = params_.output->fp32_data();
            if (out_data)
            {
                LOG_DEBUG("[RMSNormStage] output[0:8]=" << std::setprecision(6)
                                                        << out_data[0] << "," << out_data[1] << "," << out_data[2] << "," << out_data[3] << ","
                                                        << out_data[4] << "," << out_data[5] << "," << out_data[6] << "," << out_data[7]
                                                        << " device_id=" << params_.device_id.to_string());
            }
        }

        // === Stage Tracing (Task 3) ===
        if (success)
            traceOutput("output", params_.output);
        return success;
    }

    size_t RMSNormStage::estimatedFlops() const
    {
        if (!params_.input)
            return 0;

        const int seq_len = params_.seq_len > 0 ? params_.seq_len : static_cast<int>(params_.input->rows());
        const int hidden_dim = static_cast<int>(params_.input->cols());
        // Per row: hidden_dim squares + hidden_dim adds + sqrt + div + hidden_dim muls
        // Approximately 4 * hidden_dim FLOPs per row
        return static_cast<size_t>(4) * seq_len * hidden_dim;
    }

    size_t RMSNormStage::estimatedMemoryBytes() const
    {
        if (!params_.input)
            return 0;

        const int seq_len = params_.seq_len > 0 ? params_.seq_len : static_cast<int>(params_.input->rows());
        const int hidden_dim = static_cast<int>(params_.input->cols());
        // Read input + gamma, write output (in-place, so same buffer)
        size_t input_bytes = static_cast<size_t>(seq_len) * hidden_dim * sizeof(float);
        size_t gamma_bytes = static_cast<size_t>(hidden_dim) * sizeof(float);
        return 2 * input_bytes + gamma_bytes; // Read + write + gamma
    }

    bool RMSNormStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:
            return true;
#ifdef HAVE_CUDA
        case ComputeBackendType::GPU_CUDA:
            return true; // CUDARMSNormKernelT is available via KernelFactory
#endif
#ifdef HAVE_ROCM
        case ComputeBackendType::GPU_ROCM:
            return false; // ROCm RMSNorm not yet implemented
#endif
        default:
            return false;
        }
    }

    StageDumpInfo RMSNormStage::getDumpInfo() const
    {
        StageDumpInfo info;

        if (!params_.input)
            return info;

        const int seq_len = params_.seq_len > 0 ? params_.seq_len : static_cast<int>(params_.input->rows());
        const int hidden_dim = static_cast<int>(params_.input->cols());

        // Use TensorBase* overload for type-safe FP32 extraction (handles Q8_1)
        info.addInput("input", params_.input, seq_len, hidden_dim);

        // Gamma weights
        if (params_.gamma)
        {
            info.addInput("gamma", params_.gamma, 1, hidden_dim);
        }

        // Output - use TensorBase* overload
        info.addOutput("output", params_.output, seq_len, hidden_dim);

        // Scalar params
        info.addScalarInt("seq_len", seq_len);
        info.addScalarInt("hidden_dim", hidden_dim);
        info.addScalar("eps", params_.eps);

        return info;
    }

    StageBufferRequirements RMSNormStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (!params_.input)
            return reqs; // Empty if tensors not set

        // Get dimensions from tensors
        const size_t seq_len = params_.seq_len > 0
                                   ? static_cast<size_t>(params_.seq_len)
                                   : params_.input->rows();
        const size_t hidden_dim = params_.input->cols();

        // Convert tensor type to buffer tensor type
        BufferTensorType buf_type = toBufferTensorType(params_.input->native_type());

        // INPUT buffer (may be in-place with output)
        reqs.addInput("input", {seq_len, hidden_dim}, buf_type);

        // OUTPUT buffer
        if (params_.output)
        {
            reqs.addOutput("output", {seq_len, hidden_dim}, buf_type);
        }

        // WEIGHT buffer (gamma - always FP32)
        if (params_.gamma)
        {
            reqs.addWeight("gamma", {hidden_dim}, BufferTensorType::FP32);
        }

        return reqs;
    }

} // namespace llaminar2
