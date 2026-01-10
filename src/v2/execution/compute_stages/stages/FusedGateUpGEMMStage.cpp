/**
 * @file FusedGateUpGEMMStage.cpp
 * @brief Implementation of FusedGateUpGEMMStage
 */

#include "FusedGateUpGEMMStage.h"
#include "../ComputeStageUtils.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../utils/Logger.h"
#include "../../../kernels/KernelFactory.h"

namespace llaminar2
{

    // =============================================================================
    // FusedGateUpGEMMStage Implementation
    // =============================================================================

    FusedGateUpGEMMStage::FusedGateUpGEMMStage(Params params) : params_(std::move(params)) {}

    bool FusedGateUpGEMMStage::execute(IDeviceContext *ctx)
    {
        LOG_DEBUG("[FusedGateUpGEMMStage] Execute: m=" << params_.m << " k=" << params_.k
                                                       << " n_gate=" << params_.n_gate << " n_up=" << params_.n_up);

        if (!ctx)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] Null device context");
            return false;
        }

        // Validate inputs
        if (!params_.input)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] Null input");
            return false;
        }
        if (!params_.w_gate || !params_.w_up)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] Null weight tensor(s)");
            return false;
        }
        if (!params_.output_gate || !params_.output_up)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] Null output buffer(s)");
            return false;
        }
        if (params_.m <= 0 || params_.k <= 0)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] Invalid dimensions: m=" << params_.m << " k=" << params_.k);
            return false;
        }

        LOG_DEBUG("[FusedGateUpGEMMStage] input_type=" << params_.input->dtype_name()
                                                       << " output_gate_type=" << params_.output_gate->dtype_name()
                                                       << " output_up_type=" << params_.output_up->dtype_name());

        // Cast ITensor* to TensorBase* for KernelFactory
        auto *w_gate_base = requireTensorBase(params_.w_gate, "w_gate");
        auto *w_up_base = requireTensorBase(params_.w_up, "w_up");
        if (!w_gate_base || !w_up_base)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] GPU tensors not yet supported");
            return false;
        }

        // Get the target device type from device_id
        DeviceType target_dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(params_.device_id);

        // Get fused Gate/Up kernel from KernelFactory
        auto *fused_kernel = llaminar::v2::kernels::KernelFactory::getOrCreateFusedGateUpGemm(
            w_gate_base, w_up_base, target_dev_type);

        if (!fused_kernel)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] Failed to get fused Gate/Up kernel");
            return false;
        }

        // Cast input/output tensors
        auto *input_base = requireTensorBase(params_.input, "input");
        auto *output_gate_base = asTensorBase(params_.output_gate, "output_gate");
        auto *output_up_base = asTensorBase(params_.output_up, "output_up");

        if (!input_base || !output_gate_base || !output_up_base)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] Failed to cast input/output tensors");
            return false;
        }

        // Check if we have bias - use appropriate execute method
        if (params_.bias_gate || params_.bias_up)
        {
            bool success = fused_kernel->execute_with_bias(
                input_base, output_gate_base, output_up_base,
                params_.bias_gate, params_.bias_up,
                params_.m, params_.k, params_.n_gate, params_.n_up,
                ctx, params_.device_id.toKernelDeviceIndex());

            if (!success)
            {
                LOG_ERROR("[FusedGateUpGEMMStage] execute_with_bias failed");
                return false;
            }
        }
        else
        {
            bool success = fused_kernel->execute(
                input_base, output_gate_base, output_up_base,
                params_.m, params_.k, params_.n_gate, params_.n_up,
                ctx, params_.device_id.toKernelDeviceIndex());

            if (!success)
            {
                LOG_ERROR("[FusedGateUpGEMMStage] execute failed");
                return false;
            }
        }

        LOG_DEBUG("[FusedGateUpGEMMStage] Complete");
        return true;
    }

    size_t FusedGateUpGEMMStage::estimatedFlops() const
    {
        // Two GEMMs: 2 * M * N * K each
        size_t flops_gate = static_cast<size_t>(2) * params_.m * params_.n_gate * params_.k;
        size_t flops_up = static_cast<size_t>(2) * params_.m * params_.n_up * params_.k;
        return flops_gate + flops_up;
    }

    size_t FusedGateUpGEMMStage::estimatedMemoryBytes() const
    {
        // Input: m * k reads (shared)
        size_t input_bytes = static_cast<size_t>(params_.m) * params_.k * sizeof(float);

        // Outputs: gate, up writes
        size_t output_bytes = static_cast<size_t>(params_.m) * (params_.n_gate + params_.n_up) * sizeof(float);

        // Weight reads (approximate - actual depends on quantization format)
        // Quantized weights are ~4-8 bits/element, but we estimate conservatively
        size_t weight_bytes = static_cast<size_t>(params_.k) * (params_.n_gate + params_.n_up) * sizeof(float) / 4;

        return input_bytes + output_bytes + weight_bytes;
    }

    bool FusedGateUpGEMMStage::supportsBackend(ComputeBackendType backend) const
    {
        // Uses ITensorGemm interface - device-agnostic
        // Actual device support depends on the kernel implementation returned by KernelFactory
        switch (backend)
        {
        case ComputeBackendType::CPU:

        case ComputeBackendType::GPU_CUDA:
        case ComputeBackendType::GPU_ROCM:
            return true;
        default:
            return false;
        }
    }

    StageDumpInfo FusedGateUpGEMMStage::getDumpInfo() const
    {
        StageDumpInfo info;

        // Input (shared)
        info.addInput("input", params_.input, params_.m, params_.k);

        // Weight tensors
        info.addWeight("w_gate", params_.w_gate);
        info.addWeight("w_up", params_.w_up);

        // Outputs
        info.addOutput("output_gate", params_.output_gate, params_.m, params_.n_gate);
        info.addOutput("output_up", params_.output_up, params_.m, params_.n_up);

        // Scalar params
        info.addScalarInt("m", params_.m);
        info.addScalarInt("k", params_.k);
        info.addScalarInt("n_gate", params_.n_gate);
        info.addScalarInt("n_up", params_.n_up);

        return info;
    }

    StageBufferRequirements FusedGateUpGEMMStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (!params_.input || !params_.w_gate || !params_.w_up)
            return reqs; // Empty if tensors not set

        // Convert tensor types
        BufferTensorType input_type = toBufferTensorType(params_.input->native_type());
        BufferTensorType gate_type = toBufferTensorType(params_.w_gate->native_type());
        BufferTensorType up_type = toBufferTensorType(params_.w_up->native_type());

        // INPUT buffer (shared activation)
        reqs.addInput("input", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.k)}, input_type);

        // WEIGHT buffers
        reqs.addWeight("w_gate", {static_cast<size_t>(params_.k), static_cast<size_t>(params_.n_gate)}, gate_type);
        reqs.addWeight("w_up", {static_cast<size_t>(params_.k), static_cast<size_t>(params_.n_up)}, up_type);

        // OUTPUT buffers
        BufferTensorType out_type = params_.output_gate
                                        ? toBufferTensorType(params_.output_gate->native_type())
                                        : BufferTensorType::FP32;
        reqs.addOutput("output_gate", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.n_gate)}, out_type);
        reqs.addOutput("output_up", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.n_up)}, out_type);

        // Optional biases
        if (params_.bias_gate)
        {
            reqs.addWeight("bias_gate", {static_cast<size_t>(params_.n_gate)}, BufferTensorType::FP32);
        }
        if (params_.bias_up)
        {
            reqs.addWeight("bias_up", {static_cast<size_t>(params_.n_up)}, BufferTensorType::FP32);
        }

        return reqs;
    }

} // namespace llaminar2
