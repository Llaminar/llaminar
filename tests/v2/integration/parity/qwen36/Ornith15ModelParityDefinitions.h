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
 * The independent Q8 accuracy workload retains the production four-ROCm drift
 * reproducer and its CPU control without granting an HTTP certification tag.
 */
#pragma once

#include "Qwen36ModelParityDefinitions.h"
#include "Ornith15AccuracyWorkload.h"
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace llaminar2::test::parity::qwen36
{
    /**
     * @brief Return the exact no-thinking chat prefix that exposes Ornith's
     *        CUDA two-device forward-path divergence.
     *
     * This is deliberately the fully rendered tokenizer input rather than a
     * message-level approximation.  The server trace and the CPU/FP32 Hugging
     * Face oracle authenticate the resulting 39 tokens byte-for-byte.  Keeping
     * it with the typed model definition makes a failure in chat-template,
     * tokenization, graph prefill, or first-token decode independently visible
     * in the standard CSV checkpoint evidence.
     *
     * @return Exact raw prompt submitted to the model tokenizer.
     */
    inline constexpr std::string_view kOrnith15ChatNoThinkingForwardPrompt =
        R"ORNITH(<|im_start|>system
You are a calculator. Reply with only the numeric answer, no explanation.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
<think>

</think>

)ORNITH";

    /**
     * @brief Return the server-authenticated tokenization of the exact chat prompt.
     *
     * The terminal blank lines are semantic: removing them changes the Qwen
     * chat prefix and would diagnose a different request than the production
     * HTTP failure.
     *
     * @return Exact 39-token tokenizer result owned by the test declaration.
     */
    inline std::vector<int> ornith15ChatNoThinkingForwardTokenIds()
    {
        return {
            248045, 8678, 198, 2523, 513, 264, 28974, 13, 17308, 440,
            1132, 279, 23311, 4087, 11, 874, 15673, 13, 248046, 198,
            248045, 846, 198, 3710, 369, 220, 17, 10, 17, 30, 248046,
            198, 248045, 74455, 198, 248068, 271, 248069, 271,
        };
    }

    /**
     * @brief Declare the minimal checkpoint oracle for the observed HTTP bug.
     *
     * The declaration remains an ordinary typed ExpertOverlay definition: the
     * central expander supplies its Static/Dynamic and Ordinal/Random policy
     * points, while this model identity intentionally disables MTP so the
     * first native forward can be localized without speculative sidecars.
     * The focused CUDA/Static/Ordinal cell is the immediate regression target;
     * the sibling cells remain available to determine whether the defect is
     * placement- or movement-specific after the first divergent stage is known.
     *
     * @param topology Exact production CUDA ExpertOverlay topology.
     * @param reference_directory Isolated CPU/FP32 Hugging Face checkpoint pack.
     * @return Declarative one-token forward-parity matrix definition.
     */
    inline ModelParityDefinition ornith15MoEQ4ChatNoThinkingForwardParityDefinition(
        ModelParityTopologyDefinition topology,
        std::string reference_directory)
    {
        auto definition = qwen36MoEParityDefinition(
            std::move(topology), std::move(reference_directory),
            qwen36MoEExpertOverlayThresholds());
        definition.model.test_id = "Ornith15MoE_35B_Q4KM_ChatNoThinkingForward";
        definition.model.model_path =
            "/opt/llaminar-models/Ornith-1.5-35B-Q4_K_M.gguf";
        definition.model.prompt =
            std::string(kOrnith15ChatNoThinkingForwardPrompt);
        definition.model.token_ids = ornith15ChatNoThinkingForwardTokenIds();
        definition.model.decode_steps = 1;
        definition.features.mtp = ModelParityAxisProfile::Disabled;
        definition.e2e_certifiable.clear();
        return definition;
    }

    /**
     * @brief Declare exact-weight, long-decode Ornith Q8 mathematical coverage.
     * @param topology Production participant and collective identity.
     * @param reference_directory Isolated CPU/FP32 Hugging Face checkpoint pack.
     * @return Standard MTP and, for overlays, placement/movement matrix input.
     *
     * The same frozen prompt and 89-step horizon serve the CPU control and
     * failing four-ROCm topology. Existing strict checkpoint/CSV and mandatory
     * prefix-restore contracts remain intact; this diagnostic does not acquire
     * generation baselines or expand HTTP certification eligibility.
     */
    inline ModelParityDefinition ornith15MoEQ8AccuracyParityDefinition(
        ModelParityTopologyDefinition topology,
        std::string reference_directory)
    {
        const bool overlay = static_cast<bool>(topology.expert_overlay_plan);
        auto definition = qwen36MoEParityDefinition(
            std::move(topology), std::move(reference_directory),
            overlay ? qwen36MoEExpertOverlayThresholds()
                    : qwen36MoESingleDeviceThresholds());
        definition.model.test_id = "Ornith15MoE_35B_Q8_0_NaturalDecode";
        definition.model.model_path =
            "/opt/llaminar-models/Ornith-1.5-35B-Q8_0.gguf";
        definition.model.prompt = kOrnith15AccuracyPrompt;
        definition.model.token_ids = ornith15AccuracyTokenIds();
        definition.model.decode_steps = kOrnith15AccuracyDecodeSteps;
        definition.e2e_certifiable.clear();
        return definition;
    }

    /** @return The reported four-MI50 RCCL topology, with automatic capacity. */
    inline ModelParityTopologyDefinition ornith15MoERocm4ExpertOverlayTopology()
    {
        return qwen36MoEGPUExpertOverlayTopology(
            "LocalTP_RCCL_4xROCm_ExpertOverlay", "ornith15_moe_rocm_local_tp",
            Collective::RCCL, CollectiveBackendType::RCCL,
            {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1),
             GlobalDeviceAddress::rocm(2), GlobalDeviceAddress::rocm(3)});
    }

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
            // Cloud certification is explicit model eligibility, not an
            // inherited promise for a different set of fine-tuned weights.
            for (auto &selection : variant.e2e_certifiable)
                selection.remote_cpu_overlays.clear();
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
