/**
 * @file MTPModelClassPolicy.h
 * @brief Bind MTP learned-policy keys to already-loaded model expert geometry.
 *
 * Architecture names are schema identifiers, not MoE capability declarations.
 * The loaded original GGUF directory is the authority for expert cardinality
 * and selected experts, matching GraphConfig::isMoE(). This host-only projection
 * performs no file access, weight preparation, backend query, or model parsing.
 * Both execution owners consume the same resulting active request policy.
 */
#pragma once

#include "../config/RuntimeConfig.h"
#include "../../interfaces/IModelContext.h"
#include "../../loaders/ModelLoader.h"
#include <stdexcept>

namespace llaminar2
{
    /**
     * @brief Classify the admitted original model without architecture-name heuristics.
     * @param context Loaded original model authority, or null for a model-free runner.
     * @return Any for an absent model; otherwise the exact Dense or MoE capability.
     * @throws std::invalid_argument For missing source identity or incomplete/invalid expert geometry.
     *
     * A real model with no expert fields is dense even if its name contains
     * "moe". A declared expert bank must have a positive, bounded selection;
     * malformed metadata must not silently enter a dense trained-policy lane.
     */
    [[nodiscard]] inline MTPDepthPolicyModelClass resolveMTPDepthPolicyModelClass(
        const IModelContext *context)
    {
        if (!context)
            return MTPDepthPolicyModelClass::Any;
        const GGUFModel *const model = context->ggufDirectory();
        if (!model || model->architecture.empty())
            throw std::invalid_argument("MTP model-class admission requires the loaded original GGUF directory and architecture");
        const auto count = model->metadata.find(model->architecture + ".expert_count");
        const auto selected = model->metadata.find(model->architecture + ".expert_used_count");
        if (count == model->metadata.end() && selected == model->metadata.end())
            return MTPDepthPolicyModelClass::Dense;
        if (count == model->metadata.end() || selected == model->metadata.end())
            throw std::invalid_argument("MTP model-class admission requires complete expert count and selection geometry");
        const auto experts = count->second.asUInt64Checked();
        const auto used = selected->second.asUInt64Checked();
        if (experts == 0 && used == 0)
            return MTPDepthPolicyModelClass::Dense;
        if (experts == 0 || used == 0 || used > experts)
            throw std::invalid_argument("MTP model-class admission has invalid expert count or selection geometry");
        return MTPDepthPolicyModelClass::MoE;
    }

    /**
     * @brief Bind only the physical backend/model keys of an ephemeral active policy.
     * @param active Request policy after exact sampling-law specialization.
     * @param device Original continuation endpoint, not an expert offload tier.
     * @param context Loaded target model authority; null explicitly means model-free execution.
     * @return Active view with authored bounds, initialization, thresholds and physical envelope unchanged.
     * @throws std::invalid_argument If a present model lacks valid original expert metadata.
     */
    [[nodiscard]] inline MTPRuntimeConfig resolveMTPRequestDepthPolicyContext(
        MTPRuntimeConfig active, DeviceId device, const IModelContext *context)
    {
        active.depth_policy.backend = device.is_cuda() ? MTPDepthPolicyBackend::CUDA
            : device.is_rocm() ? MTPDepthPolicyBackend::ROCm
            : device.is_cpu() ? MTPDepthPolicyBackend::CPU : MTPDepthPolicyBackend::Any;
        active.depth_policy.model_class = resolveMTPDepthPolicyModelClass(context);
        return active;
    }
}
