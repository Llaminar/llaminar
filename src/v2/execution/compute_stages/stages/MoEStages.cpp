/**
 * @file MoEStages.cpp
 * @brief Implementation of MoEStages
 */

#include "MoEStages.h"
#include "../ComputeStageUtils.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../kernels/KernelFactory.h"

namespace llaminar2
{

    // =============================================================================
    // MoE Stages Implementation
    // =============================================================================

    MoERouterStage::MoERouterStage(Params params)
        : IComputeStage(params.device_id)
        , params_(std::move(params))
    {
    }

    bool MoERouterStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[MoERouterStage] Null device context");
            return false;
        }

        if (!params_.hidden || !params_.gate_weights || !params_.router_logits)
        {
            LOG_ERROR("[MoERouterStage] Null tensor parameters");
            return false;
        }

        // Cast ITensor* to TensorBase* for CPU operations
        // TODO: Add GPU support with CUDA kernels
        auto *hidden_base = dynamic_cast<const TensorBase *>(params_.hidden);
        auto *gate_weights_base = dynamic_cast<const TensorBase *>(params_.gate_weights);
        if (!hidden_base || !gate_weights_base)
        {
            LOG_ERROR("[MoERouterStage] Input tensors must be CPU TensorBase (GPU not yet supported)");
            return false;
        }

        // Get data pointers from tensors
        const float *hidden = hidden_base->data();
        const float *gate_weights = gate_weights_base->data();

        // Router logits is output, need mutable access
        auto *logits_tensor = dynamic_cast<FP32Tensor *>(params_.router_logits);
        if (!logits_tensor)
        {
            LOG_ERROR("[MoERouterStage] router_logits must be FP32Tensor");
            return false;
        }
        float *logits = logits_tensor->mutable_data();

        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int num_experts = params_.num_experts;

        ctx->runFor(0, static_cast<size_t>(seq_len), [=](size_t t_)
                    {
        int t = static_cast<int>(t_);
        const float* h = hidden + t * d_model;
        float* out = logits + t * num_experts;
        
        for (int e = 0; e < num_experts; ++e) {
            const float* w = gate_weights + e * d_model;
            float dot = 0.0f;
            for (int d = 0; d < d_model; ++d) {
                dot += h[d] * w[d];
            }
            out[e] = dot;
        } });
        return true;
    }

    size_t MoERouterStage::estimatedFlops() const
    {
        // seq_len * d_model * num_experts (dot products)
        return static_cast<size_t>(2) * params_.seq_len * params_.d_model * params_.num_experts;
    }

    bool MoERouterStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:

            return true;
        default:
            return false;
        }
    }

    StageBufferRequirements MoERouterStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (params_.hidden)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.hidden->native_type());
            reqs.addInput("hidden", params_.hidden->shape(), buf_type);
        }

        if (params_.gate_weights)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.gate_weights->native_type());
            reqs.addWeight("gate_weights", params_.gate_weights->shape(), buf_type);
        }

        if (params_.router_logits)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.router_logits->native_type());
            reqs.addOutput("router_logits", params_.router_logits->shape(), buf_type);
        }

        return reqs;
    }

    StageDumpInfo MoERouterStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        if (params_.hidden)
        {
            info.addInput("hidden", params_.hidden, params_.seq_len, params_.d_model);
        }

        if (params_.gate_weights)
        {
            info.addWeight("gate_weights", params_.gate_weights);
        }

        if (params_.router_logits)
        {
            info.addOutput("router_logits", params_.router_logits, params_.seq_len, params_.num_experts);
        }

        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarInt("num_experts", params_.num_experts);

        return info;
    }

    // -----------------------------------------------------------------------------

    MoEExpertStage::MoEExpertStage(Params params)
        : IComputeStage(params.device_id)
        , params_(std::move(params))
    {
    }

    bool MoEExpertStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[MoEExpertStage] Null device context");
            return false;
        }

        if (!params_.token_indices || params_.token_indices->empty())
        {
            // No tokens routed to this expert - nothing to do
            return true;
        }

        if (!params_.input || !params_.output)
        {
            LOG_ERROR("[MoEExpertStage] Null input or output tensor");
            return false;
        }

        // This is a placeholder - real implementation would use the actual expert weights
        // For now, we just demonstrate the structure
        LOG_DEBUG("[MoEExpertStage] Processing expert " << params_.expert_id
                                                        << " with " << params_.token_indices->size() << " tokens");

        // In real implementation:
        // 1. Gather tokens from input based on token_indices
        // 2. Apply gate projection
        // 3. Apply up projection
        // 4. SwiGLU activation
        // 5. Apply down projection
        // 6. Scatter results back

        return true;
    }

    std::string MoEExpertStage::name() const
    {
        std::ostringstream oss;
        oss << "MOE_EXPERT_" << params_.expert_id;
        return oss.str();
    }

    size_t MoEExpertStage::estimatedFlops() const
    {
        if (!params_.token_indices)
            return 0;
        size_t num_tokens = params_.token_indices->size();
        // FFN: gate + up + down projections
        // gate: num_tokens * d_model * intermediate_dim
        // up: num_tokens * d_model * intermediate_dim
        // down: num_tokens * intermediate_dim * d_model
        return static_cast<size_t>(6) * num_tokens * params_.d_model * params_.intermediate_dim;
    }

    bool MoEExpertStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:

            return true;
        default:
            return false;
        }
    }

    StageBufferRequirements MoEExpertStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (params_.input)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.input->native_type());
            reqs.addInput("input", params_.input->shape(), buf_type);
        }

        if (params_.output)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.output->native_type());
            reqs.addOutput("output", params_.output->shape(), buf_type);
        }

        // Note: Expert weights (gate, up, down) would be added here when we have
        // proper weight tensor references in the Params struct

        return reqs;
    }

    StageDumpInfo MoEExpertStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        if (params_.input)
        {
            info.addInput("input", params_.input, params_.input->rows(), params_.d_model);
        }

        if (params_.gate_w)
        {
            info.addWeight("gate_w", params_.gate_w);
        }

        if (params_.up_w)
        {
            info.addWeight("up_w", params_.up_w);
        }

        if (params_.down_w)
        {
            info.addWeight("down_w", params_.down_w);
        }

        if (params_.output)
        {
            info.addOutput("output", params_.output, params_.output->rows(), params_.d_model);
        }

        info.addScalarInt("expert_id", params_.expert_id);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarInt("intermediate_dim", params_.intermediate_dim);
        info.addScalarInt("num_tokens", params_.token_indices ? static_cast<int>(params_.token_indices->size()) : 0);

        return info;
    }

    // -----------------------------------------------------------------------------

    MoECombineStage::MoECombineStage(Params params)
        : IComputeStage(params.device_id)
        , params_(std::move(params))
    {
    }

    bool MoECombineStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[MoECombineStage] Null device context");
            return false;
        }

        // Placeholder - combines expert outputs weighted by router scores
        LOG_DEBUG("[MoECombineStage] Combining "
                  << (params_.expert_outputs ? params_.expert_outputs->size() : 0)
                  << " expert outputs");

        // In real implementation:
        // For each token:
        //   output[t] = sum over k experts: weight[t][k] * expert_output[k][t]

        return true;
    }

    bool MoECombineStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:

            return true;
        default:
            return false;
        }
    }

    StageBufferRequirements MoECombineStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        // Add expert outputs as inputs
        if (params_.expert_outputs)
        {
            for (size_t i = 0; i < params_.expert_outputs->size(); ++i)
            {
                const ITensor *expert_out = (*params_.expert_outputs)[i];
                if (expert_out)
                {
                    BufferTensorType buf_type = toBufferTensorType(expert_out->native_type());
                    std::string name = "expert_output_" + std::to_string(i);
                    reqs.addInput(name, expert_out->shape(), buf_type);
                }
            }
        }

        if (params_.output)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.output->native_type());
            reqs.addOutput("output", params_.output->shape(), buf_type);
        }

        return reqs;
    }

    StageDumpInfo MoECombineStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        // Static name storage to avoid dangling pointer issues
        // Supports up to 16 expert outputs (typical top_k is 2-8)
        static const char *expert_names[] = {
            "expert_output_0", "expert_output_1", "expert_output_2", "expert_output_3",
            "expert_output_4", "expert_output_5", "expert_output_6", "expert_output_7",
            "expert_output_8", "expert_output_9", "expert_output_10", "expert_output_11",
            "expert_output_12", "expert_output_13", "expert_output_14", "expert_output_15"};
        static constexpr size_t MAX_EXPERT_OUTPUTS = sizeof(expert_names) / sizeof(expert_names[0]);

        // Add expert outputs as inputs
        if (params_.expert_outputs)
        {
            for (size_t i = 0; i < params_.expert_outputs->size() && i < MAX_EXPERT_OUTPUTS; ++i)
            {
                const ITensor *expert_out = (*params_.expert_outputs)[i];
                if (expert_out)
                {
                    // Cast to TensorBase for addInput (requires const TensorBase*)
                    const TensorBase *expert_out_base = dynamic_cast<const TensorBase *>(expert_out);
                    if (expert_out_base)
                    {
                        info.addInput(expert_names[i], expert_out_base, expert_out->rows(), expert_out->cols());
                    }
                }
            }
        }

        if (params_.output)
        {
            // Cast to TensorBase for addOutput
            const TensorBase *output_base = dynamic_cast<const TensorBase *>(params_.output);
            if (output_base)
            {
                info.addOutput("output", output_base, params_.seq_len, params_.d_model);
            }
        }

        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarInt("top_k", params_.top_k);

        return info;
    }

} // namespace llaminar2
