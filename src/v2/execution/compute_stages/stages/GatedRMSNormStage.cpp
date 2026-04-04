/**
 * @file GatedRMSNormStage.cpp
 * @brief Implementation of gated RMS normalization stage
 */

#include "GatedRMSNormStage.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"

#include <cmath>

namespace llaminar2
{

    GatedRMSNormStage::GatedRMSNormStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool GatedRMSNormStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "GatedRMSNormStage"))
            return false;

        if (!ensureRequiredPointers("GatedRMSNormStage",
                                    {{"input", params_.input},
                                     {"gate", params_.gate},
                                     {"output", params_.output},
                                     {"gamma", const_cast<ITensor *>(params_.gamma)}}))
            return false;

        auto *input_base = requireTensorBasePtr(params_.input, "input");
        auto *gate_base = requireTensorBasePtr(params_.gate, "gate");
        auto *output_base = requireTensorBasePtr(params_.output, "output");
        auto *gamma_base = dynamic_cast<const TensorBase *>(params_.gamma);

        if (!input_base || !gate_base || !output_base || !gamma_base)
        {
            LOG_ERROR("[GatedRMSNormStage] Tensor cast failed");
            return false;
        }

        const int seq_len = (params_.seq_len > 0)
                                ? params_.seq_len
                                : static_cast<int>(input_base->rows());

        const size_t d_model = input_base->shape().size() > 1 ? input_base->shape()[1] : input_base->numel();

        const float *input_data = input_base->data();
        const float *gate_data = gate_base->data();
        const float *gamma_data = gamma_base->data();
        float *output_data = output_base->mutable_data();

        if (!input_data || !gate_data || !gamma_data || !output_data)
        {
            LOG_ERROR("[GatedRMSNormStage] Null data pointer");
            return false;
        }

        // Determine normalization dimension. When norm_dim > 0, normalize
        // over chunks of norm_dim (per-head normalization). Otherwise, full d_model.
        const size_t norm_dim = (params_.norm_dim > 0)
                                    ? static_cast<size_t>(params_.norm_dim)
                                    : d_model;
        const size_t n_groups = d_model / norm_dim; // Number of heads (or 1 for full-dim)

        for (int t = 0; t < seq_len; ++t)
        {
            const size_t row_offset = static_cast<size_t>(t) * d_model;

            for (size_t g = 0; g < n_groups; ++g)
            {
                const size_t offset = row_offset + g * norm_dim;

                // Compute RMS over norm_dim elements
                float sum_sq = 0.0f;
                for (size_t j = 0; j < norm_dim; ++j)
                {
                    const float v = input_data[offset + j];
                    sum_sq += v * v;
                }
                const float rms = std::sqrt(sum_sq / static_cast<float>(norm_dim) + params_.eps);
                const float inv_rms = 1.0f / rms;

                // Normalize, apply gamma (with optional subtract_one), multiply by gate
                for (size_t j = 0; j < norm_dim; ++j)
                {
                    const float normalized = input_data[offset + j] * inv_rms;
                    const float gamma_eff = params_.subtract_one
                                                ? (1.0f + gamma_data[j])
                                                : gamma_data[j];
                    const float gate_val = gate_data[offset + j];
                    const float gate_act = params_.gate_silu
                                               ? gate_val / (1.0f + std::exp(-gate_val))
                                               : gate_val;
                    output_data[offset + j] = normalized * gamma_eff * gate_act;
                }
            }
        }

        LOG_DEBUG("[GatedRMSNormStage] Executed: seq_len=" << seq_len
                                                           << " d_model=" << d_model
                                                           << " norm_dim=" << norm_dim
                                                           << " n_groups=" << n_groups
                                                           << " gate_silu=" << params_.gate_silu
                                                           << " subtract_one=" << params_.subtract_one);
        return true;
    }

    size_t GatedRMSNormStage::estimatedFlops() const
    {
        auto *input_base = dynamic_cast<TensorBase *>(params_.input);
        if (!input_base)
            return 0;
        // RMSNorm (~3 ops/element) + gate multiply (1 op/element)
        return input_base->numel() * 4;
    }

    size_t GatedRMSNormStage::estimatedMemoryBytes() const
    {
        auto *input_base = dynamic_cast<TensorBase *>(params_.input);
        if (!input_base)
            return 0;
        // Read input + gate + gamma, write output
        return input_base->numel() * 4 * sizeof(float);
    }

    bool GatedRMSNormStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::CPU;
    }

    StageDumpInfo GatedRMSNormStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        // Use actual seq_len dimensions, not the buffer capacity.
        // Output cols = same as input cols (norm doesn't change dimension).
        auto *out_base = dynamic_cast<const TensorBase *>(params_.output);
        if (out_base)
        {
            const size_t rows = params_.seq_len > 0
                                    ? static_cast<size_t>(params_.seq_len)
                                    : out_base->rows();
            info.addOutput("output", params_.output, rows, out_base->cols());
        }

        return info;
    }

    StageBufferRequirements GatedRMSNormStage::getBufferRequirements() const
    {
        return {};
    }

    StageBufferContract GatedRMSNormStage::bufferContract() const
    {
        StageBufferContract contract;
        if (params_.input_buffer_id)
            contract.addInput(*params_.input_buffer_id);
        if (params_.gate_buffer_id)
            contract.addInput(*params_.gate_buffer_id);
        if (params_.output_buffer_id)
            contract.addOutput(*params_.output_buffer_id);
        return contract;
    }

} // namespace llaminar2
