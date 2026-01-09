/**
 * @file FusedGateUpGEMMStage.cpp
 * @brief Implementation of FusedGateUpGEMMStage
 */

#include "FusedGateUpGEMMStage.h"
#include "../ComputeStageUtils.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../kernels/cpu/gemm_v4/FusedGEMM.h"

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
        DeviceType target_dev_type = params_.device_id.is_gpu()
                                         ? DeviceType::CUDA
                                         : DeviceType::CPU;

        // Get cached kernels from KernelFactory with device targeting
        auto *gemm_gate = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(w_gate_base, target_dev_type);
        auto *gemm_up = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(w_up_base, target_dev_type);

        if (!gemm_gate || !gemm_up)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] Failed to get GEMM kernel(s)");
            return false;
        }

        // Check if outputs are Q8_1 - if so, use multiply_tensor for type-aware dispatch
        // The fused FP32 path doesn't support Q8_1 output directly
        bool gate_is_q8_1 = (params_.output_gate->native_type() == TensorType::Q8_1);
        bool up_is_q8_1 = (params_.output_up->native_type() == TensorType::Q8_1);

        if (gate_is_q8_1 || up_is_q8_1)
        {
            LOG_DEBUG("[FusedGateUpGEMMStage] Using multiply_tensor for Q8_1 output type dispatch");

            // Cast tensors for multiply_tensor
            auto *input_base = requireTensorBase(params_.input, "input");
            auto *output_gate_base = asTensorBase(params_.output_gate, "output_gate");
            auto *output_up_base = asTensorBase(params_.output_up, "output_up");

            // For Q8_1 outputs, we need to use multiply_tensor which handles the type-aware dispatch
            // This means we lose the fusion optimization, but Q8_1 output is more important
            // TODO: Implement fused multiply_tensor_multi for Q8_1 outputs
            // TODO: multiply_tensor doesn't support bias - would need separate bias add pass
            if (params_.bias_gate || params_.bias_up)
            {
                LOG_WARN("[FusedGateUpGEMMStage] Q8_1 output path does not support bias - bias will be ignored!");
            }

            // Gate projection
            bool gate_ok = gemm_gate->multiply_tensor(
                input_base, output_gate_base,
                params_.m, params_.n_gate, params_.k,
                false,      // transpose_B
                1.0f, 0.0f, // alpha, beta
                params_.mpi_ctx, params_.device_id.toLegacyIndex());

            if (!gate_ok)
            {
                LOG_ERROR("[FusedGateUpGEMMStage] Gate GEMM multiply_tensor failed");
                return false;
            }

            // Up projection
            bool up_ok = gemm_up->multiply_tensor(
                input_base, output_up_base,
                params_.m, params_.n_up, params_.k,
                false,      // transpose_B
                1.0f, 0.0f, // alpha, beta
                params_.mpi_ctx, params_.device_id.toLegacyIndex());

            if (!up_ok)
            {
                LOG_ERROR("[FusedGateUpGEMMStage] Up GEMM multiply_tensor failed");
                return false;
            }

            LOG_DEBUG("[FusedGateUpGEMMStage] Q8_1 path complete");
            return true;
        }

        // FP32/BF16/FP16 output path - use optimized fused multiply
        const float *input_fp32 = params_.input->fp32_data();
        float *output_gate_fp32 = params_.output_gate->mutable_data();
        float *output_up_fp32 = params_.output_up->mutable_data();

        if (!input_fp32 || !output_gate_fp32 || !output_up_fp32)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] Failed to get FP32 data from tensors");
            return false;
        }

        // Build projection descriptors using device-agnostic interface
        // Pass through bias pointers from params_
        std::vector<ITensorGemm::FusedProjectionDesc> projections = {
            {gemm_gate, output_gate_fp32, params_.n_gate, params_.bias_gate, nullptr, false, "gate"},
            {gemm_up, output_up_fp32, params_.n_up, params_.bias_up, nullptr, false, "up"}};

        // Use the interface method - kernel decides if it can do optimized fusion
        bool success = gemm_gate->multiply_fused(
            input_fp32,
            projections,
            params_.m,
            params_.k);

        if (!success)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] multiply_fused failed");
            return false;
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
