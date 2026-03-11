/**
 * @file LMHeadStage.cpp
 * @brief Implementation of LMHeadStage
 */

#include "LMHeadStage.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../utils/GemmContext.h"

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
        ScopedGemmContext gemm_ctx(GemmContext::LM_HEAD);

        LOG_DEBUG("[LMHeadStage] Execute: seq_len=" << params_.seq_len
                                                    << " d_model=" << params_.d_model
                                                    << " vocab_size=" << params_.vocab_size);

        // Validate inputs
        if (!ensureRequiredPointers("LMHeadStage", {
                                                       {"hidden_states", params_.hidden_states},
                                                       {"lm_head_weight", params_.lm_head_weight},
                                                       {"logits", params_.logits},
                                                   }))
        {
            return false;
        }

        if (params_.seq_len <= 0 || params_.d_model <= 0 || params_.vocab_size <= 0)
        {
            LOG_ERROR("[LMHeadStage] Invalid dimensions");
            return false;
        }

        // Cast ITensor* to TensorBase* (works for both CPU and GPU tensor types)
        auto *hidden_states = requireTensorBasePtr(params_.hidden_states, "hidden_states");
        auto *lm_head_weight = requireTensorBasePtr(params_.lm_head_weight, "lm_head_weight");
        auto *logits = requireTensorBasePtr(params_.logits, "logits");
        if (!hidden_states || !lm_head_weight || !logits)
        {
            LOG_ERROR("[LMHeadStage] Failed to cast tensors to TensorBase");
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
        auto *prepared = llaminar::v2::kernels::KernelFactory::getOrCreatePreparedGemmWeights(lm_head_weight, params_.device_id);
        ITensorGemm *lm_gemm = llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(prepared);
        LOG_DEBUG("[LMHeadStage] Got kernel=" << (void *)lm_gemm);
        if (!lm_gemm)
        {
            LOG_ERROR("[LMHeadStage] Failed to get/create LM head GEMM kernel");
            return false;
        }
        bindStageStream(lm_gemm);

        // LM head: logits = hidden @ lm_head^T + bias
        // hidden: [seq_len, d_model], lm_head: [vocab_size, d_model]
        // output: [seq_len, vocab_size]
        //
        // OPTIMIZATION: During prefill (seq_len > 1), only compute the last token's
        // logits. The sampler only reads vocab_size elements from the last token, so
        // computing all seq_len rows is wasted. This reduces:
        //   - GEMM compute by seq_len× (e.g., 596× for typical prompts)
        //   - D2H copy from seq_len × vocab_size × 4 bytes to 1 × vocab_size × 4 bytes
        //     (e.g., 345 MB → 608 KB for Qwen2.5-7B with vocab_size=152064)
        // The result is written to row 0 of the logits tensor.
        const int lm_m = (params_.seq_len > 1) ? 1 : params_.seq_len;
        const int lm_activation_offset = (params_.seq_len > 1) ? (params_.seq_len - 1) : 0;

        // Bias is passed directly to GEMM kernel for fused application
        bool success = lm_gemm->multiply_tensor(
            hidden_states,
            logits,
            lm_m,
            params_.vocab_size,
            params_.d_model,
            true, // transpose_B (lm_head is [vocab_size, d_model])
            1.0f, 0.0f,
            params_.bias_tensor, // Bias fused into GEMM
            params_.mpi_ctx,
            params_.device_id.toKernelDeviceIndex(),
            nullptr, // workspace
            lm_activation_offset);

        if (!success)
        {
            LOG_ERROR("[LMHeadStage] GEMM failed");
            return false;
        }

        return true;
    }

    size_t LMHeadStage::estimatedFlops() const
    {
        // During prefill (seq_len > 1), only 1 row is computed
        const int effective_m = (params_.seq_len > 1) ? 1 : params_.seq_len;
        // GEMM: 2 * m * vocab_size * d_model
        return 2ULL * effective_m * params_.vocab_size * params_.d_model;
    }

    size_t LMHeadStage::estimatedMemoryBytes() const
    {
        // During prefill (seq_len > 1), only 1 row of hidden states and logits
        const int effective_m = (params_.seq_len > 1) ? 1 : params_.seq_len;
        // Read: hidden [m, d_model], lm_head [vocab_size, d_model]
        // Write: logits [m, vocab_size]
        size_t hidden_bytes = effective_m * params_.d_model * sizeof(float);
        size_t weight_bytes = params_.vocab_size * params_.d_model * sizeof(float); // Approximation for quantized
        size_t output_bytes = effective_m * params_.vocab_size * sizeof(float);
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

    StageDumpInfo LMHeadStage::buildDumpInfoImpl() const
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
            // During prefill, only 1 row of logits is computed (at row 0)
            const int effective_m = (params_.seq_len > 1) ? 1 : params_.seq_len;
            info.addOutput("logits", params_.logits,
                           effective_m, params_.vocab_size);
        }

        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarInt("vocab_size", params_.vocab_size);
        info.addScalarBool("has_bias", params_.bias_tensor != nullptr);
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

        // Optional bias tensor
        if (params_.bias_tensor)
        {
            BufferTensorType bias_type = toBufferTensorType(params_.bias_tensor->native_type());
            reqs.addWeight("bias", {static_cast<size_t>(params_.vocab_size)}, bias_type);
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
        // (vocab_size × M × sizeof(int32_t) can be large, but DeviceGraphOrchestrator
        // now dynamically sizes the budget based on max_seq_len × vocab_size)
        auto *lm_head_weight = requireTensorBasePtr(params_.lm_head_weight, "lm_head_weight");
        if (!lm_head_weight)
        {
            LOG_DEBUG("[LMHeadStage::getKernelAsWorkspaceConsumer] No weight tensor available");
            return nullptr;
        }

        auto *prepared = llaminar::v2::kernels::KernelFactory::getOrCreatePreparedGemmWeights(lm_head_weight, params_.device_id);
        ITensorGemm *lm_gemm = llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(prepared);
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

    StageBufferContract LMHeadStage::bufferContract() const
    {
        if (!params_.input_buffer_id || !params_.output_buffer_id)
            return {};

        return StageBufferContract::build()
            .addInput(*params_.input_buffer_id)
            .addOutput(*params_.output_buffer_id);
    }

} // namespace llaminar2
