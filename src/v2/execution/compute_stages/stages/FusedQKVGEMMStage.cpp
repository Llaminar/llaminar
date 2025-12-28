/**
 * @file FusedQKVGEMMStage.cpp
 * @brief Implementation of FusedQKVGEMMStage
 */

#include "FusedQKVGEMMStage.h"
#include "../ComputeStageUtils.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../kernels/cpu/gemm_v4/FusedGEMM.h"

namespace llaminar2
{

    // =============================================================================
    // FusedQKVGEMMStage Implementation
    // =============================================================================

    FusedQKVGEMMStage::FusedQKVGEMMStage(Params params) : params_(std::move(params)) {}

    bool FusedQKVGEMMStage::execute(IDeviceContext *ctx)
    {
        LOG_DEBUG("[FusedQKVGEMMStage] Execute: m=" << params_.m << " k=" << params_.k
                                                    << " n_q=" << params_.n_q << " n_k=" << params_.n_k << " n_v=" << params_.n_v);

        if (!ctx)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Null device context");
            return false;
        }

        // Validate inputs
        if (!params_.input)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Null input");
            return false;
        }
        if (!params_.wq || !params_.wk || !params_.wv)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Null weight tensor(s)");
            return false;
        }
        if (!params_.output_q || !params_.output_k || !params_.output_v)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Null output buffer(s)");
            return false;
        }
        if (params_.m <= 0 || params_.k <= 0)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Invalid dimensions: m=" << params_.m << " k=" << params_.k);
            return false;
        }

        // Check if outputs are Q8_1 tensors - if so, use Q8_1 execution path
        auto *output_q_q8_1 = dynamic_cast<Q8_1Tensor *>(params_.output_q);
        auto *output_k_q8_1 = dynamic_cast<Q8_1Tensor *>(params_.output_k);
        auto *output_v_q8_1 = dynamic_cast<Q8_1Tensor *>(params_.output_v);

        const bool q8_1_output = (output_q_q8_1 && output_k_q8_1 && output_v_q8_1);

        if (q8_1_output)
        {
            LOG_DEBUG("[FusedQKVGEMMStage] Q8_1 output detected, using Q8_1 execution path");

            // Create FusedGEMM kernel for Q8_1 output support
            FusedGEMM fused_gemm(params_.wq, params_.wk, params_.wv);

            // Check if input is also Q8_1 - use Q8_1→Q8_1 path to avoid double quantization
            auto *input_q8_1 = dynamic_cast<const Q8_1Tensor *>(params_.input);

            if (input_q8_1)
            {
                LOG_DEBUG("[FusedQKVGEMMStage] Q8_1 input detected, using Q8_1→Q8_1 path");

                // Pure Q8_1 path: Q8_1 input → Q8_1 output
                bool success = fused_gemm.execute_q8_1_to_q8_1(
                    input_q8_1->q8_1_blocks(),
                    output_q_q8_1->mutable_q8_1_blocks(),
                    output_k_q8_1->mutable_q8_1_blocks(),
                    output_v_q8_1->mutable_q8_1_blocks(),
                    params_.bias_q, params_.bias_k, params_.bias_v,
                    params_.m, params_.n_q, params_.n_k,
                    params_.k,
                    nullptr, -1); // ctx, device_idx

                if (!success)
                {
                    LOG_ERROR("[FusedQKVGEMMStage] execute_q8_1_to_q8_1 failed");
                    return false;
                }
            }
            else
            {
                LOG_DEBUG("[FusedQKVGEMMStage] FP32 input detected, using FP32→Q8_1 path");

                // FP32 input → Q8_1 output path
                const float *input_fp32 = params_.input->fp32_data();
                if (!input_fp32)
                {
                    LOG_ERROR("[FusedQKVGEMMStage] Failed to get FP32 data from input tensor");
                    return false;
                }

                bool success = fused_gemm.execute_to_q8_1(
                    input_fp32,
                    output_q_q8_1->mutable_q8_1_blocks(),
                    output_k_q8_1->mutable_q8_1_blocks(),
                    output_v_q8_1->mutable_q8_1_blocks(),
                    params_.bias_q, params_.bias_k, params_.bias_v,
                    params_.m, params_.n_q, params_.n_k,
                    params_.k,
                    nullptr, -1); // ctx, device_idx

                if (!success)
                {
                    LOG_ERROR("[FusedQKVGEMMStage] execute_to_q8_1 failed");
                    return false;
                }
            }

            LOG_DEBUG("[FusedQKVGEMMStage] Q8_1 path complete");
            return true;
        }

        // FP32 output path (original implementation)
        // Extract FP32 data from tensors - works for all tensor types via fp32_data()
        const float *input_fp32 = params_.input->fp32_data();
        float *output_q_fp32 = params_.output_q->mutable_data();
        float *output_k_fp32 = params_.output_k->mutable_data();
        float *output_v_fp32 = params_.output_v->mutable_data();

        if (!input_fp32 || !output_q_fp32 || !output_k_fp32 || !output_v_fp32)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Failed to get FP32 data from tensors");
            return false;
        }

        // DEBUG: Log input pointer and first values for parity testing
        LOG_TRACE("[FusedQKVGEMMStage] input_fp32 ptr=" << static_cast<const void *>(input_fp32)
                                                        << " input[0:4]=" << std::setprecision(10)
                                                        << input_fp32[0] << "," << input_fp32[1] << ","
                                                        << input_fp32[2] << "," << input_fp32[3]);

        LOG_DEBUG("[FusedQKVGEMMStage] input_type=" << params_.input->dtype_name()
                                                    << " output_type=" << params_.output_q->dtype_name());

        // Get cached kernels from KernelFactory (handles weight packing once)
        auto *gemm_q = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(params_.wq);
        auto *gemm_k = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(params_.wk);
        auto *gemm_v = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(params_.wv);

        if (!gemm_q || !gemm_k || !gemm_v)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Failed to get GEMM kernel(s)");
            return false;
        }

        // Build projection descriptors using device-agnostic interface
        // Pass through bias pointers from params_
        std::vector<ITensorGemm::FusedProjectionDesc> projections = {
            {gemm_q, output_q_fp32, params_.n_q, params_.bias_q, nullptr, false, "Q"},
            {gemm_k, output_k_fp32, params_.n_k, params_.bias_k, nullptr, false, "K"},
            {gemm_v, output_v_fp32, params_.n_v, params_.bias_v, nullptr, false, "V"}};

        // Use the interface method - kernel decides if it can do optimized fusion
        // Note: We use gemm_q's multiply_fused since all kernels should be same type
        bool success = gemm_q->multiply_fused(
            input_fp32,
            projections,
            params_.m,
            params_.k);

        if (!success)
        {
            LOG_ERROR("[FusedQKVGEMMStage] multiply_fused failed");
            return false;
        }

        // Debug: Log Q output for comparison
        LOG_TRACE("[FusedQKVGEMMStage] Q output[0:8]=" << std::setprecision(10)
                                                       << output_q_fp32[0] << "," << output_q_fp32[1] << "," << output_q_fp32[2] << "," << output_q_fp32[3] << ","
                                                       << output_q_fp32[4] << "," << output_q_fp32[5] << "," << output_q_fp32[6] << "," << output_q_fp32[7]
                                                       << " n_q=" << params_.n_q);

        LOG_DEBUG("[FusedQKVGEMMStage] Complete");
        return true;
    }

    size_t FusedQKVGEMMStage::estimatedFlops() const
    {
        // Three GEMMs: 2 * M * N * K each
        size_t flops_q = static_cast<size_t>(2) * params_.m * params_.n_q * params_.k;
        size_t flops_k = static_cast<size_t>(2) * params_.m * params_.n_k * params_.k;
        size_t flops_v = static_cast<size_t>(2) * params_.m * params_.n_v * params_.k;
        return flops_q + flops_k + flops_v;
    }

    size_t FusedQKVGEMMStage::estimatedMemoryBytes() const
    {
        // Input: m * k reads (shared)
        size_t input_bytes = static_cast<size_t>(params_.m) * params_.k * sizeof(float);

        // Outputs: Q, K, V writes
        size_t output_bytes = static_cast<size_t>(params_.m) * (params_.n_q + params_.n_k + params_.n_v) * sizeof(float);

        // Weight reads (approximate - actual depends on quantization format)
        // Quantized weights are ~4-8 bits/element, but we estimate conservatively
        size_t weight_bytes = static_cast<size_t>(params_.k) * (params_.n_q + params_.n_k + params_.n_v) * sizeof(float) / 4;

        return input_bytes + output_bytes + weight_bytes;
    }

    bool FusedQKVGEMMStage::supportsBackend(ComputeBackendType backend) const
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

    StageDumpInfo FusedQKVGEMMStage::getDumpInfo() const
    {
        StageDumpInfo info;

        // Input (shared)
        info.addInput("input", params_.input, params_.m, params_.k);

        // Weight tensors
        info.addWeight("wq", params_.wq);
        info.addWeight("wk", params_.wk);
        info.addWeight("wv", params_.wv);

        // Outputs
        info.addOutput("output_q", params_.output_q, params_.m, params_.n_q);
        info.addOutput("output_k", params_.output_k, params_.m, params_.n_k);
        info.addOutput("output_v", params_.output_v, params_.m, params_.n_v);

        // Scalar params
        info.addScalarInt("m", params_.m);
        info.addScalarInt("k", params_.k);
        info.addScalarInt("n_q", params_.n_q);
        info.addScalarInt("n_k", params_.n_k);
        info.addScalarInt("n_v", params_.n_v);

        return info;
    }

    StageBufferRequirements FusedQKVGEMMStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (!params_.input || !params_.wq || !params_.wk || !params_.wv)
            return reqs; // Empty if tensors not set

        // Convert tensor types
        BufferTensorType input_type = params_.input
                                          ? toBufferTensorType(params_.input->native_type())
                                          : BufferTensorType::FP32;
        BufferTensorType wq_type = toBufferTensorType(params_.wq->native_type());
        BufferTensorType wk_type = toBufferTensorType(params_.wk->native_type());
        BufferTensorType wv_type = toBufferTensorType(params_.wv->native_type());

        // INPUT buffer (shared activation)
        reqs.addInput("input", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.k)}, input_type);

        // WEIGHT buffers
        reqs.addWeight("wq", {static_cast<size_t>(params_.k), static_cast<size_t>(params_.n_q)}, wq_type);
        reqs.addWeight("wk", {static_cast<size_t>(params_.k), static_cast<size_t>(params_.n_k)}, wk_type);
        reqs.addWeight("wv", {static_cast<size_t>(params_.k), static_cast<size_t>(params_.n_v)}, wv_type);

        // OUTPUT buffers
        BufferTensorType out_type = params_.output_q
                                        ? toBufferTensorType(params_.output_q->native_type())
                                        : BufferTensorType::FP32;
        reqs.addOutput("output_q", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.n_q)}, out_type);
        reqs.addOutput("output_k", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.n_k)}, out_type);
        reqs.addOutput("output_v", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.n_v)}, out_type);

        // Optional biases
        if (params_.bias_q)
        {
            reqs.addWeight("bias_q", {static_cast<size_t>(params_.n_q)}, BufferTensorType::FP32);
        }
        if (params_.bias_k)
        {
            reqs.addWeight("bias_k", {static_cast<size_t>(params_.n_k)}, BufferTensorType::FP32);
        }
        if (params_.bias_v)
        {
            reqs.addWeight("bias_v", {static_cast<size_t>(params_.n_v)}, BufferTensorType::FP32);
        }

        return reqs;
    }


} // namespace llaminar2
