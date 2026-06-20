/**
 * @file FusedAddAllreduceStage.cpp
 * @brief Implementation of fused residual-add + TP allreduce stage
 */

#include "FusedAddAllreduceStage.h"
#include "../../../memory/StageBufferContract.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../kernels/cpu/primitives/VectorPrimitives.h"
#include "../../../utils/Logger.h"
#include "../../../utils/KernelProfiler.h"
#include "../../../utils/DebugEnv.h"

#include <cstring>

namespace llaminar2
{

    FusedAddAllreduceStage::FusedAddAllreduceStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool FusedAddAllreduceStage::execute(IDeviceContext *ctx)
    {
        KERNEL_PROFILE_SCOPE(KernelType::ALLREDUCE);

        (void)ctx;

        if (!params_.input_a || !params_.input_b || !params_.output)
        {
            LOG_ERROR("FusedAddAllreduceStage: null input/output tensor");
            return false;
        }

        if (!params_.tp_ctx)
        {
            LOG_ERROR("FusedAddAllreduceStage: null tp_ctx");
            return false;
        }

        const size_t n = (params_.num_elements > 0)
                             ? params_.num_elements
                             : params_.input_a->numel();

        // Step 1: Residual add — output = input_a + input_b (AVX-512 vectorized)
        const float *a = params_.input_a->data();
        const float *b = params_.input_b->data();
        float *out = params_.output->mutable_data();

        primitives::vec_add(out, a, b, static_cast<int>(n));

        // Step 2: Allreduce the sum in-place
        if (params_.tp_ctx->degree() <= 1)
        {
            return true; // Single device, no allreduce needed
        }

        if (debugEnv().skip_allreduce)
        {
            return true;
        }

        const size_t effective_count = (params_.allreduce_count > 0)
                                           ? params_.allreduce_count
                                           : n;

        // Use the output tensor (which now holds input_a + input_b) for in-place allreduce
        TensorBase *output_tensor = const_cast<TensorBase *>(
            dynamic_cast<const TensorBase *>(params_.output));
        if (!output_tensor)
        {
            LOG_ERROR("FusedAddAllreduceStage: output is not a TensorBase");
            return false;
        }

        bool success;
        void *stage_stream = gpuStream();
        if (!params_.stage_name.empty())
        {
            // When a GPU stream is set, route through the on-stream path
            // so the allreduce is graph-capturable and precision-aware.
            if (stage_stream)
            {
                success = params_.tp_ctx->allreduceOnStream(
                    output_tensor, params_.stage_name, effective_count, stage_stream,
                    params_.precision);
            }
            else
            {
                success = params_.tp_ctx->allreduce(output_tensor, params_.stage_name, effective_count);
            }
        }
        else
        {
            success = params_.tp_ctx->allreduce(output_tensor);
        }

        if (!success)
        {
            LOG_ERROR("FusedAddAllreduceStage: allreduce failed");
            return false;
        }

        return true;
    }

    size_t FusedAddAllreduceStage::estimatedFlops() const
    {
        const size_t n = (params_.num_elements > 0)
                             ? params_.num_elements
                             : (params_.input_a ? params_.input_a->numel() : 0);
        return n; // One addition per element
    }

    bool FusedAddAllreduceStage::supportsBackend(ComputeBackendType backend) const
    {
        // Supports all backends (add is trivial, allreduce via TP context)
        (void)backend;
        return true;
    }

    StageDumpInfo FusedAddAllreduceStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        const size_t n = (params_.num_elements > 0)
                             ? params_.num_elements
                             : (params_.input_a ? params_.input_a->numel() : 0);
        const int cols = params_.output ? static_cast<int>(params_.output->shape().back()) : 0;
        const int rows = cols > 0 ? static_cast<int>(n / cols) : 0;

        if (params_.input_a)
            info.addInput("input_a", params_.input_a, rows, cols);
        if (params_.input_b)
            info.addInput("input_b", params_.input_b, rows, cols);
        if (params_.output)
            info.addOutput("output", params_.output, rows, cols);
        return info;
    }

    StageBufferContract FusedAddAllreduceStage::bufferContract() const
    {
        StageBufferContract contract;
        if (params_.input_a_buffer_id)
            contract.addInput(*params_.input_a_buffer_id);
        if (params_.input_b_buffer_id)
            contract.addInput(*params_.input_b_buffer_id);
        if (params_.output_buffer_id)
            contract.addOutput(*params_.output_buffer_id);
        return contract;
    }

} // namespace llaminar2
