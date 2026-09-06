/**
 * @file Qwen38ModelParityDefinitions.h
 * @brief Dense Qwen3.8 real-weight identity and canonical certification policy.
 *
 * The installed IQ4_XS GGUF declares qwen35, 65 blocks including one nextn
 * predictor, 24 query heads and four KV heads. It uses the existing hybrid-GDN
 * production/reference family, but never reuses a Qwen3.6 reference identity.
 */
#pragma once

#include "../ModelParityDefinition.h"
#include "../qwen36/Qwen36MTPCheckpointSurface.h"

namespace llaminar2::test::parity::qwen38
{
    /** @return Model-authenticated dense 27B IQ4_XS definition for one topology. */
    inline ModelParityDefinition qwen38DenseParityDefinition(
        ModelParityTopologyDefinition topology, std::string reference_directory)
    {
        ModelParityDefinition definition;
        definition.model = {
            .test_id = "Qwen38Dense_27B_IQ4XS",
            .model_path = "/opt/llaminar-models/Qwen3.8-27B-IQ4_XS.gguf",
            .reference_directory = std::move(reference_directory),
            .prompt = "The quick brown fox jumps over the lazy dog",
            .decode_steps = 3,
            .max_seq_len = 4096,
            .transformer_layers = 64,
            .attention_heads = 24,
            .kv_heads = 4,
            .maximum_mtp_draft_depth = kModelParityRequiredMaximumMTPDepth,
            .mtp_checkpoint_surface = qwen36::qwen36DenseMTPCheckpointSurface(),
        };
        definition.topology = std::move(topology);
        // Retain the prior dense gate until fresh CSV evidence proves this
        // model/quant. A new filename does not justify weaker tolerances.
        definition.thresholds = {
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.93f,
            .early_layers_count = 8,
            .min_early_layers_passed = 8,
            .kl_threshold = 0.08f,
            .min_top1_accuracy = 80.0f,
            .min_top5_accuracy = 80.0f,
            .pytorch_top1_in_topk = 3,
        };
        definition.precisions.activation = {ActivationPrecision::FP32};
        definition.precisions.kv_cache = {KVCachePrecision::FP16};
        definition.features.mtp = ModelParityAxisProfile::Standard;
        // Initial HTTP certification covers adaptive MTP on each GPU vendor.
        // CPU and fixed-depth cells retain their full mathematical obligations.
        if (definition.topology.kind == ModelParityTopologyKind::SingleDevice &&
            definition.topology.participants.size() == 1 &&
            !definition.topology.participants.front().address.isCPU())
            definition.e2e_certifiable = {{.mtp = ModelParityMTP::DynamicDepth}};
        return definition;
    }
} // namespace llaminar2::test::parity::qwen38
