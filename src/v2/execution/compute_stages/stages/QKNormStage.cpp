/**
 * @file QKNormStage.cpp
 * @brief Implementation of QKNormStage (per-head QK RMSNorm for Qwen3)
 */

#include "QKNormStage.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../kernels/KernelFactory.h"

namespace llaminar2
{

    QKNormStage::QKNormStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool QKNormStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "QKNormStage"))
        {
            return false;
        }

        if (!ensureRequiredPointers("QKNormStage", {
                                                       {"input", params_.input},
                                                       {"output", params_.output},
                                                       {"gamma", params_.gamma},
                                                   }))
        {
            return false;
        }

        auto *input_base = requireTensorBasePtr(params_.input, "input");
        auto *gamma_base = requireTensorBasePtr(params_.gamma, "gamma");
        auto *output_base = requireTensorBasePtr(params_.output, "output");
        if (!input_base || !gamma_base || !output_base)
        {
            LOG_ERROR("[QKNormStage] GPU tensors not yet supported");
            return false;
        }

        traceInput("input", params_.input);
        traceInput("gamma", params_.gamma);

        // Remap dimensions for per-head normalization:
        //   Physical:  [seq_len, n_heads * head_dim]
        //   Logical:   [seq_len * n_heads, head_dim]
        const int total_tokens = params_.seq_len > 0
                                     ? params_.seq_len
                                     : static_cast<int>(input_base->rows());
        const int effective_rows = total_tokens * params_.n_heads;
        const int hidden_dim = params_.head_dim;

        LOG_DEBUG("[QKNormStage] Execute: total_tokens=" << total_tokens
                                                         << " n_heads=" << params_.n_heads
                                                         << " head_dim=" << params_.head_dim
                                                         << " effective_rows=" << effective_rows
                                                         << " hidden_dim=" << hidden_dim
                                                         << " eps=" << params_.eps
                                                         << " tensor_type=" << input_base->dtype_name()
                                                         << " device=" << params_.device_id.to_string());

        // Reuse the stage-local RMSNorm kernel unless the input tensor dtype changes.
        auto *kernel = getOrRefreshKernelByTensorType(
            cached_kernel_,
            cached_kernel_tensor_type_,
            input_base,
            [&]()
            {
                return llaminar::v2::kernels::KernelFactory::getOrCreateRMSNorm(
                    input_base, params_.device_id);
            });
        if (!kernel)
        {
            LOG_ERROR("[QKNormStage] Failed to create RMSNorm kernel for type "
                      << input_base->dtype_name());
            return false;
        }

        // Thread GPU stream for graph capture
        bindStageStream(kernel);

        // Call the RMSNorm kernel with remapped dimensions:
        //   seq_len = total_tokens * n_heads (each head is a separate "row")
        //   hidden_dim = head_dim (normalize across head dimension)
        bool success = kernel->apply_tensor(
            input_base,
            gamma_base,
            output_base,
            effective_rows,
            hidden_dim,
            params_.eps,
            params_.mpi_ctx,
            params_.device_id.toKernelDeviceIndex());

        if (success)
            traceOutput("output", params_.output);
        return success;
    }

    size_t QKNormStage::estimatedFlops() const
    {
        if (!params_.input)
            return 0;

        const int total_tokens = params_.seq_len > 0
                                     ? params_.seq_len
                                     : static_cast<int>(params_.input->rows());
        // Per head: head_dim squares + head_dim adds + sqrt + div + head_dim muls
        // ~4 * head_dim FLOPs per head per token
        return static_cast<size_t>(4) * total_tokens * params_.n_heads * params_.head_dim;
    }

    size_t QKNormStage::estimatedMemoryBytes() const
    {
        if (!params_.input)
            return 0;

        const int total_tokens = params_.seq_len > 0
                                     ? params_.seq_len
                                     : static_cast<int>(params_.input->rows());
        size_t tensor_bytes = static_cast<size_t>(total_tokens) * params_.n_heads *
                              params_.head_dim * sizeof(float);
        size_t gamma_bytes = static_cast<size_t>(params_.head_dim) * sizeof(float);
        return 2 * tensor_bytes + gamma_bytes; // Read + write + gamma
    }

    bool QKNormStage::supportsBackend(ComputeBackendType backend) const
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

    StageDumpInfo QKNormStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        if (!params_.input)
            return info;

        const int total_tokens = params_.seq_len > 0
                                     ? params_.seq_len
                                     : static_cast<int>(params_.input->rows());
        const int total_cols = params_.n_heads * params_.head_dim;

        info.addInput("input", params_.input, total_tokens, total_cols);

        if (params_.gamma)
        {
            info.addInput("gamma", params_.gamma, 1, params_.head_dim);
        }

        info.addOutput("output", params_.output, total_tokens, total_cols);

        info.addScalarInt("seq_len", total_tokens);
        info.addScalarInt("n_heads", params_.n_heads);
        info.addScalarInt("head_dim", params_.head_dim);
        info.addScalar("eps", params_.eps);

        return info;
    }

    StageBufferRequirements QKNormStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (!params_.input)
            return reqs;

        const size_t seq_len = params_.seq_len > 0
                                   ? static_cast<size_t>(params_.seq_len)
                                   : params_.input->rows();
        const size_t total_cols = static_cast<size_t>(params_.n_heads) * params_.head_dim;

        BufferTensorType buf_type = toBufferTensorType(params_.input->native_type());

        reqs.addInput("input", {seq_len, total_cols}, buf_type);

        if (params_.output)
        {
            reqs.addOutput("output", {seq_len, total_cols}, buf_type);
        }

        if (params_.gamma)
        {
            reqs.addWeight("gamma", {static_cast<size_t>(params_.head_dim)}, BufferTensorType::FP32);
        }

        return reqs;
    }

    StageBufferContract QKNormStage::bufferContract() const
    {
        if (!params_.input_buffer_id || !params_.output_buffer_id)
            return {};

        auto contract = StageBufferContract::build();

        if (*params_.input_buffer_id == *params_.output_buffer_id)
        {
            contract.addInOut(*params_.input_buffer_id);
        }
        else
        {
            contract.addInput(*params_.input_buffer_id);
            contract.addOutput(*params_.output_buffer_id);
        }

        return contract;
    }

} // namespace llaminar2
