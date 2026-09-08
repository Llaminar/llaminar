/**
 * @file Ornith15ModelParityDefinitions.h
 * @brief Ornith 1.5 real-weight coverage inherited from certified Qwen3.6 MoE topologies.
 *
 * The local Q4_K_M fine-tune declares qwen35moe with forty main blocks and
 * one next-token predictor. It shares topology and feature contracts, never
 * reference weights or artifact identity, with its Qwen3.6 MoE parent. Only
 * selected topology is duplicated as a complete numerical matrix before the
 * ordinary E2E selector chooses its exact production configuration. The
 * fine-tune's larger weights require the two-CUDA overlay instead of one CUDA;
 * ROCm covers both one device and the two-device overlay, alongside CPU NodeTP.
 */
#pragma once

#include "Qwen36ModelParityDefinitions.h"
#include <span>

namespace llaminar2::test::parity::qwen36
{
    /**
     * @brief Add Ornith's single-ROCm and overlay certification definitions.
     * @param definitions Canonical Qwen3.6 topology/feature definitions.
     * @return Original definitions followed by independent Ornith identities.
     *
     * Copying the typed definition preserves budgets, topology, precision and
     * prefix obligations without another topology table. Overlay definitions
     * select Dynamic/Ordinal/adaptive MTP, including the two-GPU topologies
     * that the parent model exercises numerically but does not tag for HTTP.
     */
    inline std::vector<ModelParityDefinition> withOrnith15CertificationModels(
        std::span<const ModelParityDefinition> definitions)
    {
        std::vector<ModelParityDefinition> result(definitions.begin(), definitions.end());
        for (const auto &source : definitions)
        {
            const bool overlay = static_cast<bool>(source.topology.expert_overlay_plan);
            const bool single_rocm =
                source.topology.kind == ModelParityTopologyKind::SingleDevice &&
                source.topology.participants.size() == 1 &&
                source.topology.participants.front().address.isROCm();
            if (!overlay && !single_rocm) continue;
            auto variant = source;
            variant.model.test_id = "Ornith15MoE_35B_Q4KM";
            variant.model.model_path = "/opt/llaminar-models/Ornith-1.5-35B-Q4_K_M.gguf";
            // Never authenticate the fine-tune against its parent's reference pack.
            variant.model.reference_directory =
                "pytorch_ornith15_moe_" + source.topology.test_id + "_snapshots";
            if (overlay)
                variant.e2e_certifiable = {{
                    .mtp = ModelParityMTP::DynamicDepth,
                    .owner_order = RoutedExpertOwnerOrder::Ordinal,
                    .movement = ModelParityExpertMovement::Dynamic,
                }};
            result.push_back(std::move(variant));
        }
        return result;
    }
}
