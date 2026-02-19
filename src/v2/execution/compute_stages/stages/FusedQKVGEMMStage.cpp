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

#include <cmath>
#include <cstring>
#include <vector>

namespace llaminar2
{

    // =============================================================================
    // FusedQKVGEMMStage Implementation
    // =============================================================================

    FusedQKVGEMMStage::FusedQKVGEMMStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool FusedQKVGEMMStage::execute(IDeviceContext *ctx)
    {
        LOG_DEBUG("[FusedQKVGEMMStage] Execute: m=" << params_.m << " k=" << params_.k
                                                    << " n_q=" << params_.n_q << " n_k=" << params_.n_k << " n_v=" << params_.n_v
                                                    << " device=" << params_.device_id.to_string());

        // Log bias tensor pointers for multi-GPU debugging
        if (params_.bias_q || params_.bias_k || params_.bias_v)
        {
            LOG_DEBUG("[FusedQKVGEMMStage] BIAS POINTERS:"
                      << " bias_q=" << static_cast<const void *>(params_.bias_q ? params_.bias_q->raw_data() : nullptr)
                      << " bias_k=" << static_cast<const void *>(params_.bias_k ? params_.bias_k->raw_data() : nullptr)
                      << " bias_v=" << static_cast<const void *>(params_.bias_v ? params_.bias_v->raw_data() : nullptr)
                      << " stage_device=" << params_.device_id.to_string());
        }

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

            // Cast weights to TensorBase for KernelFactory
            auto *wq_base = requireTensorBase(params_.wq, "wq");
            auto *wk_base = requireTensorBase(params_.wk, "wk");
            auto *wv_base = requireTensorBase(params_.wv, "wv");

            // Get cached fused QKV GEMM kernel from KernelFactory
            auto *fused_kernel = llaminar::v2::kernels::KernelFactory::getOrCreateFusedQKVGemm(
                wq_base, wk_base, wv_base, params_.device_id);
            if (!fused_kernel)
            {
                LOG_ERROR("[FusedQKVGEMMStage] Failed to get fused QKV kernel");
                return false;
            }
            fused_kernel->setGPUStream(gpuStream());

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
                bool success = fused_kernel->execute_q8_1(
                    input_q8_1->typed_data(),
                    output_q_q8_1->mutable_typed_data(),
                    output_k_q16_1->mutable_typed_data(),
                    output_v_q8_1->mutable_typed_data(),
                    params_.bias_q, params_.bias_k, params_.bias_v,
                    params_.m, params_.n_q, params_.n_k,
                    params_.k,
                    k_block_size);

                if (!success)
                {
                    LOG_ERROR("[FusedQKVGEMMStage] execute_q8_1 failed");
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
            // Need to get buffer size from params
            size_t buffer_size = static_cast<size_t>(params_.m) * static_cast<size_t>(params_.k) / 32 * sizeof(Q8_1Block);

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

            // Now use the Q8_1 mixed path via the cached kernel
            bool success = fused_kernel->execute_q8_1(
                q8_1_buffer.data(),
                output_q_q8_1->mutable_typed_data(),
                output_k_q16_1->mutable_typed_data(),
                output_v_q8_1->mutable_typed_data(),
                params_.bias_q, params_.bias_k, params_.bias_v,
                params_.m, params_.n_q, params_.n_k,
                params_.k,
                k_block_size);

            if (!success)
            {
                LOG_ERROR("[FusedQKVGEMMStage] execute_q8_1 failed (FP32 input)");
                return false;
            }

            LOG_DEBUG("[FusedQKVGEMMStage] Mixed-precision QKV path complete (FP32→Q8_1 input)");
            return true;
        }

        const bool q8_1_output = (output_q_q8_1 && output_k_q8_1 && output_v_q8_1);

        if (q8_1_output)
        {
            LOG_DEBUG("[FusedQKVGEMMStage] Q8_1 output detected, using Q8_1 execution path");

            // Cast weights to TensorBase for KernelFactory
            auto *wq_base2 = requireTensorBase(params_.wq, "wq");
            auto *wk_base2 = requireTensorBase(params_.wk, "wk");
            auto *wv_base2 = requireTensorBase(params_.wv, "wv");

            // Get cached fused QKV GEMM kernel from KernelFactory
            auto *fused_kernel = llaminar::v2::kernels::KernelFactory::getOrCreateFusedQKVGemm(
                wq_base2, wk_base2, wv_base2, params_.device_id);
            if (!fused_kernel)
            {
                LOG_ERROR("[FusedQKVGEMMStage] Failed to get fused QKV kernel for Q8_1 output");
                return false;
            }
            fused_kernel->setGPUStream(gpuStream());

            // Check if input is also Q8_1 - use Q8_1→Q8_1 path to avoid double quantization
            auto *input_q8_1 = dynamic_cast<const Q8_1Tensor *>(params_.input);

            if (input_q8_1)
            {
                LOG_DEBUG("[FusedQKVGEMMStage] Q8_1 input detected, using Q8_1→Q8_1 path");

                // Pure Q8_1 path: Q8_1 input → Q8_1 output
                bool success = fused_kernel->execute_q8_1_to_q8_1(
                    input_q8_1->typed_data(),
                    output_q_q8_1->mutable_typed_data(),
                    output_k_q8_1->mutable_typed_data(),
                    output_v_q8_1->mutable_typed_data(),
                    params_.bias_q, params_.bias_k, params_.bias_v,
                    params_.m, params_.n_q, params_.n_k,
                    params_.k);

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

                bool success = fused_kernel->execute_fp32(
                    input_fp32,
                    output_q_q8_1->mutable_typed_data(),
                    output_k_q8_1->mutable_typed_data(),
                    output_v_q8_1->mutable_typed_data(),
                    params_.bias_q, params_.bias_k, params_.bias_v,
                    params_.m, params_.n_q, params_.n_k,
                    params_.k,
                    64); // default block size

                if (!success)
                {
                    LOG_ERROR("[FusedQKVGEMMStage] execute_fp32 failed");
                    return false;
                }
            }

            LOG_DEBUG("[FusedQKVGEMMStage] Q8_1 path complete");
            return true;
        }

        // Cast weights to TensorBase for KernelFactory
        auto *wq_base = requireTensorBase(params_.wq, "wq");
        auto *wk_base = requireTensorBase(params_.wk, "wk");
        auto *wv_base = requireTensorBase(params_.wv, "wv");

        // Get the target device type from device_id
        DeviceType target_dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(params_.device_id);

        // Get cached kernels from KernelFactory with device targeting
        // Use DeviceId overload to ensure correct GPU ordinal is used (e.g., ROCm:1 vs ROCm:0)
        auto *prepared_q = llaminar::v2::kernels::KernelFactory::getOrCreatePreparedGemmWeights(wq_base, params_.device_id);
        auto *prepared_k = llaminar::v2::kernels::KernelFactory::getOrCreatePreparedGemmWeights(wk_base, params_.device_id);
        auto *prepared_v = llaminar::v2::kernels::KernelFactory::getOrCreatePreparedGemmWeights(wv_base, params_.device_id);
        auto *gemm_q = llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(prepared_q);
        auto *gemm_k = llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(prepared_k);
        auto *gemm_v = llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(prepared_v);
        if (gemm_q)
            gemm_q->setGPUStream(gpuStream());
        if (gemm_k)
            gemm_k->setGPUStream(gpuStream());
        if (gemm_v)
            gemm_v->setGPUStream(gpuStream());
        const bool gpu_execution = (target_dev_type == DeviceType::CUDA || target_dev_type == DeviceType::ROCm);
        LOG_DEBUG("[FusedQKVGEMMStage] device_id=" << params_.device_id.to_string()
                                                   << " is_gpu=" << gpu_execution);
        bool success = false;

        if (gpu_execution)
        {
            // GPU path: Use tensor-aware API - kernel handles device placement
            LOG_DEBUG("[FusedQKVGEMMStage] Using tensor-aware GPU path");

            // Cast ITensor* to TensorBase* for tensor-aware API
            auto *input_base = dynamic_cast<TensorBase *>(const_cast<ITensor *>(params_.input));
            auto *output_q_base = dynamic_cast<TensorBase *>(params_.output_q);
            auto *output_k_base = dynamic_cast<TensorBase *>(params_.output_k);
            auto *output_v_base = dynamic_cast<TensorBase *>(params_.output_v);

            if (!input_base || !output_q_base || !output_k_base || !output_v_base)
            {
                LOG_ERROR("[FusedQKVGEMMStage] GPU path requires TensorBase-derived types");
                return false;
            }

            // Build tensor projection descriptors
            std::vector<ITensorGemm::TensorProjectionDesc> projections = {
                {gemm_q, output_q_base, params_.n_q, params_.bias_q, nullptr, false, "Q"},
                {gemm_k, output_k_base, params_.n_k, params_.bias_k, nullptr, false, "K"},
                {gemm_v, output_v_base, params_.n_v, params_.bias_v, nullptr, false, "V"}};

            // Use tensor-aware fused API - handles all device sync internally
            success = gemm_q->multiply_fused_tensor(
                input_base,
                projections,
                params_.m,
                params_.k);

            if (success)
            {
                LOG_DEBUG("[FusedQKVGEMMStage] GPU execution complete");
            }
        }
        else
        {
            // CPU path: Use raw pointer API
            const float *input_fp32 = params_.input->fp32_data();
            float *output_q_fp32 = params_.output_q->mutable_data();
            float *output_k_fp32 = params_.output_k->mutable_data();
            float *output_v_fp32 = params_.output_v->mutable_data();

            if (!input_fp32 || !output_q_fp32 || !output_k_fp32 || !output_v_fp32)
            {
                LOG_ERROR("[FusedQKVGEMMStage] Failed to get FP32 data from tensors");
                return false;
            }

            LOG_DEBUG("[FusedQKVGEMMStage] input_type=" << params_.input->dtype_name()
                                                        << " output_type=" << params_.output_q->dtype_name());

            // Build projection descriptors for raw pointer API
            // FusedProjectionDesc takes TensorBase* for bias directly
            std::vector<ITensorGemm::FusedProjectionDesc> projections = {
                {gemm_q, output_q_fp32, params_.n_q, params_.bias_q, nullptr, false, "Q"},
                {gemm_k, output_k_fp32, params_.n_k, params_.bias_k, nullptr, false, "K"},
                {gemm_v, output_v_fp32, params_.n_v, params_.bias_v, nullptr, false, "V"}};

            success = gemm_q->multiply_fused(
                input_fp32,
                projections,
                params_.m,
                params_.k);

            if (success)
            {
                // Debug: Log Q output for comparison
                LOG_TRACE("[FusedQKVGEMMStage] Q output[0:8]=" << std::setprecision(10)
                                                               << output_q_fp32[0] << "," << output_q_fp32[1] << ","
                                                               << output_q_fp32[2] << "," << output_q_fp32[3] << ","
                                                               << output_q_fp32[4] << "," << output_q_fp32[5] << ","
                                                               << output_q_fp32[6] << "," << output_q_fp32[7]
                                                               << " n_q=" << params_.n_q);
            }
        }

        if (!success)
        {
            LOG_ERROR("[FusedQKVGEMMStage] GEMM failed");
            return false;
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

    StageDumpInfo FusedQKVGEMMStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        // Input (shared)
        info.addInput("input", params_.input, params_.m, params_.k);

        // Weight tensors
        info.addWeight("wq", params_.wq);
        info.addWeight("wk", params_.wk);
        info.addWeight("wv", params_.wv);

        // Bias tensors (optional but needed for coherence)
        if (params_.bias_q)
        {
            info.addWeight("bias_q", params_.bias_q);
        }
        if (params_.bias_k)
        {
            info.addWeight("bias_k", params_.bias_k);
        }
        if (params_.bias_v)
        {
            info.addWeight("bias_v", params_.bias_v);
        }

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

    // =============================================================================
    // IWorkspaceConsumerStage Implementation
    // =============================================================================

    IWorkspaceConsumer *FusedQKVGEMMStage::getKernelAsWorkspaceConsumer()
    {
        // Get kernel from KernelFactory for the Q projection weight
        // All three projections share the same workspace, so returning any one works.
        // The Q kernel is always present, so use that.
        if (!params_.wq)
        {
            LOG_WARN("[FusedQKVGEMMStage::getKernelAsWorkspaceConsumer] Q weight tensor not set");
            return nullptr;
        }

        auto *wq_base = dynamic_cast<TensorBase *>(const_cast<ITensor *>(params_.wq));
        if (!wq_base)
        {
            LOG_WARN("[FusedQKVGEMMStage::getKernelAsWorkspaceConsumer] Q weight tensor is not TensorBase");
            return nullptr;
        }

        auto *prepared = llaminar::v2::kernels::KernelFactory::getOrCreatePreparedGemmWeights(wq_base, params_.device_id);
        ITensorGemm *gemm = llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(prepared);

        return dynamic_cast<IWorkspaceConsumer *>(gemm);
    }

    void FusedQKVGEMMStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        // Bind workspace to ALL THREE kernels (Q, K, V)
        // Each kernel needs workspace for GPU execution

        // Q kernel
        if (params_.wq)
        {
            auto *wq_base = dynamic_cast<TensorBase *>(const_cast<ITensor *>(params_.wq));
            if (wq_base)
            {
                auto *prepared_q = llaminar::v2::kernels::KernelFactory::getOrCreatePreparedGemmWeights(wq_base, params_.device_id);
                ITensorGemm *gemm_q = llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(prepared_q);
                if (auto *consumer_q = dynamic_cast<IWorkspaceConsumer *>(gemm_q))
                {
                    consumer_q->bindWorkspace(workspace);
                    LOG_DEBUG("[FusedQKVGEMMStage] Bound workspace to Q kernel");
                }
            }
        }

        // K kernel
        if (params_.wk)
        {
            auto *wk_base = dynamic_cast<TensorBase *>(const_cast<ITensor *>(params_.wk));
            if (wk_base)
            {
                auto *prepared_k = llaminar::v2::kernels::KernelFactory::getOrCreatePreparedGemmWeights(wk_base, params_.device_id);
                ITensorGemm *gemm_k = llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(prepared_k);
                if (auto *consumer_k = dynamic_cast<IWorkspaceConsumer *>(gemm_k))
                {
                    consumer_k->bindWorkspace(workspace);
                    LOG_DEBUG("[FusedQKVGEMMStage] Bound workspace to K kernel");
                }
            }
        }

        // V kernel
        if (params_.wv)
        {
            auto *wv_base = dynamic_cast<TensorBase *>(const_cast<ITensor *>(params_.wv));
            if (wv_base)
            {
                auto *prepared_v = llaminar::v2::kernels::KernelFactory::getOrCreatePreparedGemmWeights(wv_base, params_.device_id);
                ITensorGemm *gemm_v = llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(prepared_v);
                if (auto *consumer_v = dynamic_cast<IWorkspaceConsumer *>(gemm_v))
                {
                    consumer_v->bindWorkspace(workspace);
                    LOG_DEBUG("[FusedQKVGEMMStage] Bound workspace to V kernel");
                }
            }
        }

        // Store workspace reference for hasWorkspace()/getWorkspace()
        bound_workspace_ = workspace;
    }

    void FusedQKVGEMMStage::unbindWorkspace()
    {
        // Unbind workspace from all three kernels
        if (params_.wq)
        {
            auto *wq_base = dynamic_cast<TensorBase *>(const_cast<ITensor *>(params_.wq));
            if (wq_base)
            {
                auto *prepared_q = llaminar::v2::kernels::KernelFactory::getOrCreatePreparedGemmWeights(wq_base, params_.device_id);
                ITensorGemm *gemm_q = llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(prepared_q);
                if (auto *consumer_q = dynamic_cast<IWorkspaceConsumer *>(gemm_q))
                {
                    consumer_q->unbindWorkspace();
                }
            }
        }

        if (params_.wk)
        {
            auto *wk_base = dynamic_cast<TensorBase *>(const_cast<ITensor *>(params_.wk));
            if (wk_base)
            {
                auto *prepared_k = llaminar::v2::kernels::KernelFactory::getOrCreatePreparedGemmWeights(wk_base, params_.device_id);
                ITensorGemm *gemm_k = llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(prepared_k);
                if (auto *consumer_k = dynamic_cast<IWorkspaceConsumer *>(gemm_k))
                {
                    consumer_k->unbindWorkspace();
                }
            }
        }

        if (params_.wv)
        {
            auto *wv_base = dynamic_cast<TensorBase *>(const_cast<ITensor *>(params_.wv));
            if (wv_base)
            {
                auto *prepared_v = llaminar::v2::kernels::KernelFactory::getOrCreatePreparedGemmWeights(wv_base, params_.device_id);
                ITensorGemm *gemm_v = llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(prepared_v);
                if (auto *consumer_v = dynamic_cast<IWorkspaceConsumer *>(gemm_v))
                {
                    consumer_v->unbindWorkspace();
                }
            }
        }

        bound_workspace_ = nullptr;
    }

} // namespace llaminar2
