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
#include "../../../utils/GemmContext.h"

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
        ScopedGemmContext gemm_ctx(GemmContext::ATTN);

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

        // Cast weights to TensorBase for KernelFactory
        auto *wq_base = requireTensorBase(params_.wq, "wq");
        auto *wk_base = requireTensorBase(params_.wk, "wk");
        auto *wv_base = requireTensorBase(params_.wv, "wv");

        // Get the target device type from device_id
        DeviceType target_dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(params_.device_id);

        // Get or cache individual Q/K/V kernels (avoid KernelFactory mutex per token)
        if (!cache_resolved_individual_)
        {
            auto *prepared_q = llaminar::v2::kernels::KernelFactory::getOrCreatePreparedGemmWeights(wq_base, params_.device_id);
            auto *prepared_k = llaminar::v2::kernels::KernelFactory::getOrCreatePreparedGemmWeights(wk_base, params_.device_id);
            auto *prepared_v = llaminar::v2::kernels::KernelFactory::getOrCreatePreparedGemmWeights(wv_base, params_.device_id);
            cached_gemm_q_ = llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(prepared_q);
            cached_gemm_k_ = llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(prepared_k);
            cached_gemm_v_ = llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(prepared_v);
            cache_resolved_individual_ = true;
        }
        auto *gemm_q = cached_gemm_q_;
        auto *gemm_k = cached_gemm_k_;
        auto *gemm_v = cached_gemm_v_;
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
                {gemm_q, output_q_base, params_.n_q, params_.bias_q, "Q"},
                {gemm_k, output_k_base, params_.n_k, params_.bias_k, "K"},
                {gemm_v, output_v_base, params_.n_v, params_.bias_v, "V"}};

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
            // CPU path: Use tensor-aware API (same as GPU path)
            auto *input_base = dynamic_cast<TensorBase *>(const_cast<ITensor *>(params_.input));
            auto *output_q_base = dynamic_cast<TensorBase *>(params_.output_q);
            auto *output_k_base = dynamic_cast<TensorBase *>(params_.output_k);
            auto *output_v_base = dynamic_cast<TensorBase *>(params_.output_v);

            if (!input_base || !output_q_base || !output_k_base || !output_v_base)
            {
                LOG_ERROR("[FusedQKVGEMMStage] CPU path requires TensorBase-derived types");
                return false;
            }

            LOG_DEBUG("[FusedQKVGEMMStage] input_type=" << params_.input->dtype_name()
                                                        << " output_type=" << params_.output_q->dtype_name());

            // Build tensor projection descriptors
            std::vector<ITensorGemm::TensorProjectionDesc> projections = {
                {gemm_q, output_q_base, params_.n_q, params_.bias_q, "Q"},
                {gemm_k, output_k_base, params_.n_k, params_.bias_k, "K"},
                {gemm_v, output_v_base, params_.n_v, params_.bias_v, "V"}};

            success = gemm_q->multiply_fused_tensor(
                input_base,
                projections,
                params_.m,
                params_.k);

            if (success)
            {
                const float *q_data = output_q_base->data();
                LOG_TRACE("[FusedQKVGEMMStage] Q output[0:8]=" << std::setprecision(10)
                                                               << q_data[0] << "," << q_data[1] << ","
                                                               << q_data[2] << "," << q_data[3] << ","
                                                               << q_data[4] << "," << q_data[5] << ","
                                                               << q_data[6] << "," << q_data[7]
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

        // Use cached kernel if available, otherwise resolve
        if (!cache_resolved_individual_)
        {
            auto *prepared = llaminar::v2::kernels::KernelFactory::getOrCreatePreparedGemmWeights(wq_base, params_.device_id);
            cached_gemm_q_ = llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(prepared);
            // Note: K/V will be resolved on first execute() or bindWorkspace()
        }

        return dynamic_cast<IWorkspaceConsumer *>(cached_gemm_q_);
    }

    void FusedQKVGEMMStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        // Bind workspace to ALL THREE kernels (Q, K, V)
        // Each kernel needs workspace for GPU execution
        // Resolve and cache individual kernels if not already done
        if (!cache_resolved_individual_)
        {
            if (params_.wq)
            {
                auto *wq_base = dynamic_cast<TensorBase *>(const_cast<ITensor *>(params_.wq));
                if (wq_base)
                {
                    auto *prepared_q = llaminar::v2::kernels::KernelFactory::getOrCreatePreparedGemmWeights(wq_base, params_.device_id);
                    cached_gemm_q_ = llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(prepared_q);
                }
            }
            if (params_.wk)
            {
                auto *wk_base = dynamic_cast<TensorBase *>(const_cast<ITensor *>(params_.wk));
                if (wk_base)
                {
                    auto *prepared_k = llaminar::v2::kernels::KernelFactory::getOrCreatePreparedGemmWeights(wk_base, params_.device_id);
                    cached_gemm_k_ = llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(prepared_k);
                }
            }
            if (params_.wv)
            {
                auto *wv_base = dynamic_cast<TensorBase *>(const_cast<ITensor *>(params_.wv));
                if (wv_base)
                {
                    auto *prepared_v = llaminar::v2::kernels::KernelFactory::getOrCreatePreparedGemmWeights(wv_base, params_.device_id);
                    cached_gemm_v_ = llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(prepared_v);
                }
            }
            cache_resolved_individual_ = true;
        }

        // Bind workspace using cached kernels
        if (auto *consumer_q = dynamic_cast<IWorkspaceConsumer *>(cached_gemm_q_))
        {
            consumer_q->bindWorkspace(workspace);
            LOG_DEBUG("[FusedQKVGEMMStage] Bound workspace to Q kernel");
        }
        if (auto *consumer_k = dynamic_cast<IWorkspaceConsumer *>(cached_gemm_k_))
        {
            consumer_k->bindWorkspace(workspace);
            LOG_DEBUG("[FusedQKVGEMMStage] Bound workspace to K kernel");
        }
        if (auto *consumer_v = dynamic_cast<IWorkspaceConsumer *>(cached_gemm_v_))
        {
            consumer_v->bindWorkspace(workspace);
            LOG_DEBUG("[FusedQKVGEMMStage] Bound workspace to V kernel");
        }

        // Store workspace reference for hasWorkspace()/getWorkspace()
        bound_workspace_ = workspace;
    }

    void FusedQKVGEMMStage::unbindWorkspace()
    {
        // Unbind workspace from cached kernels
        if (auto *consumer_q = dynamic_cast<IWorkspaceConsumer *>(cached_gemm_q_))
            consumer_q->unbindWorkspace();
        if (auto *consumer_k = dynamic_cast<IWorkspaceConsumer *>(cached_gemm_k_))
            consumer_k->unbindWorkspace();
        if (auto *consumer_v = dynamic_cast<IWorkspaceConsumer *>(cached_gemm_v_))
            consumer_v->unbindWorkspace();

        bound_workspace_ = nullptr;
    }

    StageBufferContract FusedQKVGEMMStage::bufferContract() const
    {
        if (!params_.input_buffer_id || !params_.output_q_buffer_id ||
            !params_.output_k_buffer_id || !params_.output_v_buffer_id)
            return {};

        return StageBufferContract::build()
            .addInput(*params_.input_buffer_id)
            .addOutput(*params_.output_q_buffer_id)
            .addOutput(*params_.output_k_buffer_id)
            .addOutput(*params_.output_v_buffer_id);
    }

} // namespace llaminar2
