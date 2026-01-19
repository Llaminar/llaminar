/**
 * @file LMHeadStage.cpp
 * @brief Implementation of LMHeadStage
 */

#include "LMHeadStage.h"
#include "../ComputeStageUtils.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../kernels/KernelFactory.h"
#include <omp.h>

namespace llaminar2
{

    // =============================================================================
    // LMHeadStage Implementation
    // =============================================================================

    LMHeadStage::LMHeadStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool LMHeadStage::execute(IDeviceContext *ctx)
    {
        LOG_DEBUG("[LMHeadStage] Execute: seq_len=" << params_.seq_len
                                                    << " d_model=" << params_.d_model
                                                    << " vocab_size=" << params_.vocab_size);

        // Validate inputs
        if (!params_.hidden_states || !params_.lm_head_weight || !params_.logits)
        {
            LOG_ERROR("[LMHeadStage] Null tensor pointers");
            return false;
        }

        if (params_.seq_len <= 0 || params_.d_model <= 0 || params_.vocab_size <= 0)
        {
            LOG_ERROR("[LMHeadStage] Invalid dimensions");
            return false;
        }

        // Cast ITensor* to TensorBase* for CPU operations
        auto *hidden_states = requireTensorBase(params_.hidden_states, "hidden_states");
        auto *lm_head_weight = requireTensorBase(params_.lm_head_weight, "lm_head_weight");
        auto *logits = requireTensorBase(params_.logits, "logits");
        if (!hidden_states || !lm_head_weight || !logits)
        {
            LOG_ERROR("[LMHeadStage] GPU tensors not yet supported");
            return false;
        }

        // Get or create GEMM kernel for LM head weight
        // CRITICAL: Use DeviceId overload (not DeviceType) to ensure correct GPU ordinal
        // Using DeviceType alone would default to ROCm:0 even when running on ROCm:1
        LOG_DEBUG("[LMHeadStage] Requesting kernel with device_id=" << params_.device_id.toKernelDeviceIndex()
                                                                    << " (is_rocm=" << params_.device_id.is_rocm()
                                                                    << ", rocm_ordinal=" << (params_.device_id.is_rocm() ? params_.device_id.rocm_ordinal() : -1) << ")"
                                                                    << " lm_head_weight=" << (void *)lm_head_weight
                                                                    << " shape=" << lm_head_weight->shape()[0] << "x" << lm_head_weight->shape()[1]);
        ITensorGemm *lm_gemm = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(lm_head_weight, params_.device_id);
        LOG_DEBUG("[LMHeadStage] Got kernel=" << (void *)lm_gemm);
        if (!lm_gemm)
        {
            LOG_ERROR("[LMHeadStage] Failed to get/create LM head GEMM kernel");
            return false;
        }

        // LM head: logits = hidden @ lm_head^T
        // hidden: [seq_len, d_model], lm_head: [vocab_size, d_model]
        // output: [seq_len, vocab_size]
        bool success = lm_gemm->multiply_tensor(
            hidden_states,
            logits,
            params_.seq_len,
            params_.vocab_size,
            params_.d_model,
            true, // transpose_B (lm_head is [vocab_size, d_model])
            1.0f, 0.0f,
            params_.mpi_ctx,
            params_.device_id.toKernelDeviceIndex());

        if (!success)
        {
            LOG_ERROR("[LMHeadStage] GEMM failed");
            return false;
        }

        // Apply bias if present
        if (params_.bias)
        {
            float *logits_data = logits->mutable_data();
            if (!logits_data)
            {
                LOG_ERROR("[LMHeadStage] Failed to get logits data for bias add");
                return false;
            }

// Add bias to each row
#pragma omp parallel for
            for (int s = 0; s < params_.seq_len; ++s)
            {
                float *row = logits_data + s * params_.vocab_size;
                for (int v = 0; v < params_.vocab_size; ++v)
                {
                    row[v] += params_.bias[v];
                }
            }
        }

        return true;
    }

    size_t LMHeadStage::estimatedFlops() const
    {
        // GEMM: 2 * seq_len * vocab_size * d_model
        return 2ULL * params_.seq_len * params_.vocab_size * params_.d_model;
    }

    size_t LMHeadStage::estimatedMemoryBytes() const
    {
        // Read: hidden [seq_len, d_model], lm_head [vocab_size, d_model]
        // Write: logits [seq_len, vocab_size]
        size_t hidden_bytes = params_.seq_len * params_.d_model * sizeof(float);
        size_t weight_bytes = params_.vocab_size * params_.d_model * sizeof(float); // Approximation for quantized
        size_t output_bytes = params_.seq_len * params_.vocab_size * sizeof(float);
        return hidden_bytes + weight_bytes + output_bytes;
    }

    bool LMHeadStage::supportsBackend(ComputeBackendType backend) const
    {
        // LM head uses KernelFactory which supports CPU and GPU
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

    StageDumpInfo LMHeadStage::getDumpInfo() const
    {
        StageDumpInfo info;

        if (params_.hidden_states)
        {
            info.addInput("hidden_states", params_.hidden_states,
                          params_.seq_len, params_.d_model);
        }

        if (params_.lm_head_weight)
        {
            info.addWeight("lm_head_weight", params_.lm_head_weight);
        }

        if (params_.logits)
        {
            info.addOutput("logits", params_.logits,
                           params_.seq_len, params_.vocab_size);
        }

        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarInt("vocab_size", params_.vocab_size);
        info.addScalarBool("has_bias", params_.bias != nullptr);
        info.addScalarInt("device_id", params_.device_id.toKernelDeviceIndex());

        return info;
    }

    StageBufferRequirements LMHeadStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (!params_.hidden_states || !params_.lm_head_weight || !params_.logits)
            return reqs; // Empty if tensors not set

        // INPUT buffer (hidden states)
        BufferTensorType hidden_type = toBufferTensorType(params_.hidden_states->native_type());
        reqs.addInput("hidden_states",
                      {static_cast<size_t>(params_.seq_len), static_cast<size_t>(params_.d_model)},
                      hidden_type);

        // WEIGHT buffer (LM head weights - may be quantized)
        BufferTensorType weight_type = toBufferTensorType(params_.lm_head_weight->native_type());
        reqs.addWeight("lm_head_weight",
                       {static_cast<size_t>(params_.vocab_size), static_cast<size_t>(params_.d_model)},
                       weight_type);

        // OUTPUT buffer (logits)
        BufferTensorType logits_type = toBufferTensorType(params_.logits->native_type());
        reqs.addOutput("logits",
                       {static_cast<size_t>(params_.seq_len), static_cast<size_t>(params_.vocab_size)},
                       logits_type);

        // Optional bias
        if (params_.bias)
        {
            reqs.addWeight("bias", {static_cast<size_t>(params_.vocab_size)}, BufferTensorType::FP32);
        }

        return reqs;
    }

    // =============================================================================
    // IWorkspaceConsumerStage Implementation
    // =============================================================================

    IWorkspaceConsumer *LMHeadStage::getKernelAsWorkspaceConsumer()
    {
        // Get the GEMM kernel and return it as workspace consumer
        // The kernel needs workspace for its ACC_INT32 accumulator buffer
        // (vocab_size × M × sizeof(int32_t) can be large, but GraphOrchestrator
        // now dynamically sizes the budget based on max_seq_len × vocab_size)
        auto *lm_head_weight = requireTensorBase(params_.lm_head_weight, "lm_head_weight");
        if (!lm_head_weight)
        {
            LOG_DEBUG("[LMHeadStage::getKernelAsWorkspaceConsumer] No weight tensor available");
            return nullptr;
        }

        ITensorGemm *lm_gemm = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(lm_head_weight, params_.device_id);
        if (!lm_gemm)
        {
            LOG_DEBUG("[LMHeadStage::getKernelAsWorkspaceConsumer] No GEMM kernel available");
            return nullptr;
        }

        auto *consumer = dynamic_cast<IWorkspaceConsumer *>(lm_gemm);
        LOG_DEBUG("[LMHeadStage::getKernelAsWorkspaceConsumer] Returning kernel=" << (void *)lm_gemm
                                                                                  << " as consumer=" << (void *)consumer);
        return consumer;
    }

} // namespace llaminar2
