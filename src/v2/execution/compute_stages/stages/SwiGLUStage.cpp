/**
 * @file SwiGLUStage.cpp
 * @brief Implementation of SwiGLUStage
 */

#include "SwiGLUStage.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/KernelProfiler.h"
#include "../../../kernels/KernelFactory.h"

namespace llaminar2
{

    // =============================================================================
    // SwiGLUStage Implementation (Type-Safe via TensorBase)
    // =============================================================================

    SwiGLUStage::SwiGLUStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool SwiGLUStage::execute(IDeviceContext *ctx)
    {
        KERNEL_PROFILE_SCOPE(KernelType::SWIGLU);

        if (!ensureContext(ctx, "SwiGLUStage"))
        {
            return false;
        }

        if (!ensureRequiredPointers("SwiGLUStage", {
                                                   {"gate", params_.gate},
                                                   {"up", params_.up},
                                                   {"output", params_.output},
                                               }))
        {
            return false;
        }

        // Cast ITensor* to TensorBase* for CPU operations
        auto *gate_base = requireTensorBasePtr(params_.gate, "gate");
        auto *up_base = requireTensorBasePtr(params_.up, "up");
        auto *output_base = requireTensorBasePtr(params_.output, "output");
        if (!gate_base || !up_base || !output_base)
        {
            LOG_ERROR("[SwiGLUStage] GPU tensors not yet supported");
            return false;
        }

        // Use explicit seq_len if provided, otherwise derive from tensor shape
        // This is critical for decode mode where buffers are pre-allocated for max_seq_len
        // but we're only processing 1 token.
        const int seq_len = (params_.seq_len > 0)
                                ? params_.seq_len
                                : static_cast<int>(gate_base->rows());
        const int intermediate_dim = static_cast<int>(gate_base->cols());

        LOG_DEBUG("[SwiGLUStage] Execute: seq_len=" << seq_len
                                                    << " (params_.seq_len=" << params_.seq_len << ")"
                                                    << " intermediate_dim=" << intermediate_dim
                                                    << " tensor_type=" << gate_base->dtype_name()
                                                    << " device_id=" << params_.device_id.to_string());

        // Create kernel via KernelFactory with automatic type dispatch (cached)
        auto *kernel = getOrRefreshKernelByTensorType(
            cached_kernel_,
            cached_kernel_tensor_type_,
            gate_base,
            [&]()
            {
                return llaminar::v2::kernels::KernelFactory::getOrCreateSwiGLU(gate_base, params_.device_id);
            });
        if (!kernel)
        {
            LOG_ERROR("[SwiGLUStage] Failed to create SwiGLU kernel for type "
                      << gate_base->dtype_name());
            return false;
        }

        // Thread GPU stream for graph capture
        bindStageStream(kernel);

        // Apply SwiGLU via kernel's apply_tensor method
        return kernel->apply_tensor(
            gate_base,
            up_base,
            output_base,
            seq_len,
            intermediate_dim,
            false, // add_residual
            params_.mpi_ctx,
            params_.device_id.toKernelDeviceIndex());
    }

    size_t SwiGLUStage::estimatedFlops() const
    {
        if (!params_.gate)
            return 0;

        const int seq_len = static_cast<int>(params_.gate->rows());
        const int intermediate_dim = static_cast<int>(params_.gate->cols());
        // Per element: exp, div, mul, mul (~6 FLOPs)
        return static_cast<size_t>(6) * seq_len * intermediate_dim;
    }

    size_t SwiGLUStage::estimatedMemoryBytes() const
    {
        if (!params_.gate)
            return 0;

        const int seq_len = static_cast<int>(params_.gate->rows());
        const int intermediate_dim = static_cast<int>(params_.gate->cols());
        size_t bytes = static_cast<size_t>(seq_len) * intermediate_dim * sizeof(float);
        return 3 * bytes; // gate + up + output
    }

    bool SwiGLUStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:

            return true;
        default:
            return false;
        }
    }

    StageDumpInfo SwiGLUStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        if (!params_.gate || !params_.up || !params_.output)
            return info;

        // Use explicit seq_len if provided, otherwise derive from tensor
        const int seq_len = (params_.seq_len > 0) ? params_.seq_len : static_cast<int>(params_.gate->rows());
        const int intermediate_dim = static_cast<int>(params_.gate->cols());

        // Use TensorBase* overload for type-safe FP32 extraction (handles Q8_1)
        info.addInput("gate", params_.gate, seq_len, intermediate_dim);
        info.addInput("up", params_.up, seq_len, intermediate_dim);

        // Output - use TensorBase* overload
        info.addOutput("output", params_.output, seq_len, intermediate_dim);

        // Scalar params
        info.addScalarInt("seq_len", seq_len);
        info.addScalarInt("intermediate_dim", intermediate_dim);

        return info;
    }

    StageBufferRequirements SwiGLUStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (!params_.gate || !params_.up || !params_.output)
            return reqs; // Empty if tensors not set

        // Get dimensions from tensors
        const size_t rows = params_.gate->rows();
        const size_t cols = params_.gate->cols();

        // Convert tensor type to buffer tensor type
        BufferTensorType buf_type = toBufferTensorType(params_.gate->native_type());

        // INPUT buffers (read-only)
        reqs.addInput("gate", {rows, cols}, buf_type);
        reqs.addInput("up", {rows, cols}, buf_type);

        // OUTPUT buffer
        reqs.addOutput("output", {rows, cols}, buf_type);

        return reqs;
    }

} // namespace llaminar2
