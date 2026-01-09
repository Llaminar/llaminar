/**
 * @file FusedQKVGEMMStage.cpp
 * @brief Implementation of FusedQKVGEMMStage
 */

#include "FusedQKVGEMMStage.h"
#include "../ComputeStageUtils.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/FP16Utils.h"
#include "../../../utils/Logger.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../kernels/cpu/gemm_v4/FusedGEMM.h"

#include <cmath>
#include <cstring>
#include <vector>

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

        // Check output tensor types for precision detection
        auto *output_q_q8_1 = dynamic_cast<Q8_1Tensor *>(params_.output_q);
        auto *output_k_q8_1 = dynamic_cast<Q8_1Tensor *>(params_.output_k);
        auto *output_k_q16_1 = dynamic_cast<Q16_1Tensor *>(params_.output_k);
        auto *output_v_q8_1 = dynamic_cast<Q8_1Tensor *>(params_.output_v);

        // Check for mixed-precision QKV: Q=Q8_1, K=Q16_1, V=Q8_1 (HybridQ16 K precision fix)
        const bool mixed_qkv = (output_q_q8_1 && output_k_q16_1 && output_v_q8_1);

        if (mixed_qkv)
        {
            LOG_DEBUG("[FusedQKVGEMMStage] Mixed-precision QKV detected: Q=Q8_1, K=Q16_1, V=Q8_1");

            // Cast weights to TensorBase for FusedGEMM
            auto *wq_base = requireTensorBase(params_.wq, "wq");
            auto *wk_base = requireTensorBase(params_.wk, "wk");
            auto *wv_base = requireTensorBase(params_.wv, "wv");

            // Create FusedGEMM kernel
            FusedGEMM fused_gemm(wq_base, wk_base, wv_base);

            // Determine Q16 block size from K output tensor
            // LIMITATION: JIT kernel only properly supports block_size=64 or 32.
            // block_size=128 writes only partial data (64 values) per block.
            // Force block_size=64 for now until JIT kernel supports 128.
            int k_block_size = 64;

            LOG_DEBUG("[FusedQKVGEMMStage] K Q16_1 block_size=" << k_block_size);

            // Check if input is Q8_1 - use direct path
            auto *input_q8_1 = dynamic_cast<const Q8_1Tensor *>(params_.input);
            if (input_q8_1)
            {
                LOG_DEBUG("[FusedQKVGEMMStage] Q8_1 input, using direct Q8_1→mixed path");
                bool success = fused_gemm.execute_q8_1_mixed_qkv(
                    input_q8_1->typed_data(),
                    output_q_q8_1->mutable_typed_data(),
                    output_k_q16_1->mutable_typed_data(),
                    output_v_q8_1->mutable_typed_data(),
                    params_.bias_q, params_.bias_k, params_.bias_v,
                    params_.m, params_.n_q, params_.n_k,
                    params_.k,
                    k_block_size,
                    nullptr, -1); // ctx, device_idx

                if (!success)
                {
                    LOG_ERROR("[FusedQKVGEMMStage] execute_q8_1_mixed_qkv failed");
                    return false;
                }

                LOG_DEBUG("[FusedQKVGEMMStage] Mixed-precision QKV path complete (Q8_1 input)");
                return true;
            }

            // FP32 input - quantize to Q8_1 first, then use mixed path
            const float *input_fp32 = params_.input->fp32_data();
            if (!input_fp32)
            {
                LOG_ERROR("[FusedQKVGEMMStage] Mixed-precision QKV requires Q8_1 or FP32 input");
                return false;
            }

            LOG_DEBUG("[FusedQKVGEMMStage] FP32 input, quantizing to Q8_1 for mixed path");

            // Allocate temporary Q8_1 buffer for quantized activations
            // Use the first kernel to get buffer size
            size_t buffer_size = fused_gemm.num_projections() > 0
                                     ? static_cast<size_t>(params_.m) * static_cast<size_t>(params_.k) / 32 * sizeof(Q8_1Block)
                                     : 0;

            if (buffer_size == 0)
            {
                LOG_ERROR("[FusedQKVGEMMStage] Failed to compute Q8_1 buffer size");
                return false;
            }

            // Allocate aligned buffer for quantized activations
            std::vector<Q8_1Block> q8_1_buffer(params_.m * params_.k / 32);

            // Quantize FP32 input to Q8_1
            // Use scalar quantization for simplicity (GEMM kernel will handle the rest)
            for (int row = 0; row < params_.m; ++row)
            {
                const float *row_data = input_fp32 + row * params_.k;
                Q8_1Block *row_blocks = q8_1_buffer.data() + row * (params_.k / 32);

                for (int block_idx = 0; block_idx < params_.k / 32; ++block_idx)
                {
                    const float *block_data = row_data + block_idx * 32;
                    Q8_1Block &block = row_blocks[block_idx];

                    // Find max absolute value for scaling
                    float max_abs = 0.0f;
                    for (int i = 0; i < 32; ++i)
                    {
                        float abs_val = std::abs(block_data[i]);
                        if (abs_val > max_abs)
                            max_abs = abs_val;
                    }

                    // Compute scale (FP16) and sum_qs (INT16)
                    float scale_fp32 = max_abs / 127.0f;
                    block.d = fp32_to_fp16(scale_fp32); // FP16 scale
                    int32_t sum = 0;

                    if (max_abs > 0.0f)
                    {
                        float inv_scale = 127.0f / max_abs;
                        for (int i = 0; i < 32; ++i)
                        {
                            int8_t q = static_cast<int8_t>(std::round(block_data[i] * inv_scale));
                            block.qs[i] = q;
                            sum += static_cast<int32_t>(q);
                        }
                    }
                    else
                    {
                        std::memset(block.qs, 0, 32);
                    }

                    // Clamp sum to INT16 range (should be within [-127*32, 127*32] = [-4064, 4064])
                    block.sum_qs = static_cast<int16_t>(std::max(-32768, std::min(32767, sum)));
                }
            }

            LOG_DEBUG("[FusedQKVGEMMStage] Quantized FP32 to Q8_1: " << params_.m << "x" << params_.k);

            // Now use the Q8_1 mixed path
            bool success = fused_gemm.execute_q8_1_mixed_qkv(
                q8_1_buffer.data(),
                output_q_q8_1->mutable_typed_data(),
                output_k_q16_1->mutable_typed_data(),
                output_v_q8_1->mutable_typed_data(),
                params_.bias_q, params_.bias_k, params_.bias_v,
                params_.m, params_.n_q, params_.n_k,
                params_.k,
                k_block_size,
                nullptr, -1); // ctx, device_idx

            if (!success)
            {
                LOG_ERROR("[FusedQKVGEMMStage] execute_q8_1_mixed_qkv failed (FP32 input)");
                return false;
            }

            LOG_DEBUG("[FusedQKVGEMMStage] Mixed-precision QKV path complete (FP32→Q8_1 input)");
            return true;
        }

        const bool q8_1_output = (output_q_q8_1 && output_k_q8_1 && output_v_q8_1);

        if (q8_1_output)
        {
            LOG_DEBUG("[FusedQKVGEMMStage] Q8_1 output detected, using Q8_1 execution path");

            // Cast weights to TensorBase for FusedGEMM
            auto *wq_base2 = requireTensorBase(params_.wq, "wq");
            auto *wk_base2 = requireTensorBase(params_.wk, "wk");
            auto *wv_base2 = requireTensorBase(params_.wv, "wv");

            // Create FusedGEMM kernel for Q8_1 output support
            FusedGEMM fused_gemm(wq_base2, wk_base2, wv_base2);

            // Check if input is also Q8_1 - use Q8_1→Q8_1 path to avoid double quantization
            auto *input_q8_1 = dynamic_cast<const Q8_1Tensor *>(params_.input);

            if (input_q8_1)
            {
                LOG_DEBUG("[FusedQKVGEMMStage] Q8_1 input detected, using Q8_1→Q8_1 path");

                // Pure Q8_1 path: Q8_1 input → Q8_1 output
                bool success = fused_gemm.execute_q8_1_to_q8_1(
                    input_q8_1->typed_data(),
                    output_q_q8_1->mutable_typed_data(),
                    output_k_q8_1->mutable_typed_data(),
                    output_v_q8_1->mutable_typed_data(),
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
                    output_q_q8_1->mutable_typed_data(),
                    output_k_q8_1->mutable_typed_data(),
                    output_v_q8_1->mutable_typed_data(),
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
        // NOTE: For GPU execution, we need device pointers
        const float *input_fp32 = nullptr;
        float *output_q_fp32 = nullptr;
        float *output_k_fp32 = nullptr;
        float *output_v_fp32 = nullptr;

        // Cast to FP32Tensor to access GPU methods
        auto *input_t = dynamic_cast<FP32Tensor *>(const_cast<ITensor *>(params_.input));
        auto *output_q_t = dynamic_cast<FP32Tensor *>(params_.output_q);
        auto *output_k_t = dynamic_cast<FP32Tensor *>(params_.output_k);
        auto *output_v_t = dynamic_cast<FP32Tensor *>(params_.output_v);

        const bool gpu_execution = params_.device_id.is_gpu();

        if (gpu_execution)
        {
            // GPU path: Use device pointers
            if (!input_t || !output_q_t || !output_k_t || !output_v_t)
            {
                LOG_ERROR("[FusedQKVGEMMStage] GPU execution requires FP32Tensor types");
                return false;
            }

            // Ensure input is synced to device
            if (!input_t->ensureOnDevice(params_.device_id))
            {
                LOG_ERROR("[FusedQKVGEMMStage] Failed to ensure input on device " << params_.device_id.toString());
                return false;
            }

            // Ensure output buffers are allocated on device
            if (!output_q_t->ensureOnDevice(params_.device_id) ||
                !output_k_t->ensureOnDevice(params_.device_id) ||
                !output_v_t->ensureOnDevice(params_.device_id))
            {
                LOG_ERROR("[FusedQKVGEMMStage] Failed to ensure outputs on device " << params_.device_id.toString());
                return false;
            }

            // Get device pointers
            input_fp32 = static_cast<const float *>(input_t->gpu_data_ptr());
            output_q_fp32 = static_cast<float *>(output_q_t->gpu_data_ptr());
            output_k_fp32 = static_cast<float *>(output_k_t->gpu_data_ptr());
            output_v_fp32 = static_cast<float *>(output_v_t->gpu_data_ptr());

            LOG_DEBUG("[FusedQKVGEMMStage] GPU execution using device pointers (" << params_.device_id.toString() << ")");
        }
        else
        {
            // CPU path: Use host pointers
            input_fp32 = params_.input->fp32_data();
            output_q_fp32 = params_.output_q->mutable_data();
            output_k_fp32 = params_.output_k->mutable_data();
            output_v_fp32 = params_.output_v->mutable_data();
        }

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

        // Cast weights to TensorBase for KernelFactory
        auto *wq_base_fp32 = requireTensorBase(params_.wq, "wq");
        auto *wk_base_fp32 = requireTensorBase(params_.wk, "wk");
        auto *wv_base_fp32 = requireTensorBase(params_.wv, "wv");

        // Get the target device type from device_id
        DeviceType target_dev_type = params_.device_id.is_gpu()
                                         ? DeviceType::CUDA
                                         : DeviceType::CPU;

        // Get cached kernels from KernelFactory with device targeting
        auto *gemm_q = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(wq_base_fp32, target_dev_type);
        auto *gemm_k = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(wk_base_fp32, target_dev_type);
        auto *gemm_v = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(wv_base_fp32, target_dev_type);

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

        // After GPU execution, mark outputs as device-dirty
        // This tells the tensor that device has latest data, host is stale
        if (gpu_execution)
        {
            output_q_t->mark_device_dirty();
            output_k_t->mark_device_dirty();
            output_v_t->mark_device_dirty();
            LOG_DEBUG("[FusedQKVGEMMStage] Marked Q/K/V outputs as device-dirty");
        }

        // Debug: Log Q output for comparison (only for CPU - device memory not accessible)
        if (!gpu_execution)
        {
            LOG_TRACE("[FusedQKVGEMMStage] Q output[0:8]=" << std::setprecision(10)
                                                           << output_q_fp32[0] << "," << output_q_fp32[1] << "," << output_q_fp32[2] << "," << output_q_fp32[3] << ","
                                                           << output_q_fp32[4] << "," << output_q_fp32[5] << "," << output_q_fp32[6] << "," << output_q_fp32[7]
                                                           << " n_q=" << params_.n_q);
        }

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
