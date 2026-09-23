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
            .prefix_state = ModelParityPrefixState::HybridRecurrent,
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
        // Each declared GPU topology certifies adaptive MTP through automatic
        // frontend selection; fixed-depth cases remain mathematical diagnostics.
        if (!definition.topology.participants.empty() &&
            std::all_of(definition.topology.participants.begin(), definition.topology.participants.end(),
                [](const auto &participant) { return !participant.address.isCPU(); }))
        {
            // Exercise the selected admitted single-device context for each
            // certification GPU. The 24-GiB CUDA configuration requires an
            // 8K profile to leave space for this IQ4_XS model's complete
            // captured physical BOM; the 32-GiB MI50 admits 32K. Keep this
            // capability fact in the typed profile, not in the HTTP runner.
            ModelParityE2EProfile profile;
            if (definition.topology.kind == ModelParityTopologyKind::SingleDevice &&
                definition.topology.participants.front().address.isROCm())
            {
                profile.context_length = 32768;
                profile.minimum_prompt_tokens = 16384;
            }
            definition.e2e_certifiable = {{
                .mtp = ModelParityMTP::DynamicDepth,
                .profile = profile,
            }};
        }
        return definition;
    }

    /**
     * @return Dense homogeneous TP/PP and mixed-vendor TP-over-PP declarations.
     *
     * These are the canonical diagnostic topologies as well as the authority
     * for HTTP cardinality/strategy constraints. E2E never prescribes these
     * ordinal IDs or the mathematical fixture's equal layer partition to auto.
     */
    inline std::vector<ModelParityDefinition> qwen38DenseMultiDeviceDefinitions()
    {
        std::vector<ModelParityDefinition> definitions;
        for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
            for (const bool pipeline : {false, true})
            {
                const bool cuda = backend == DeviceType::CUDA;
                const std::string id = std::string(cuda ? "CUDA2" : "ROCm2") + (pipeline ? "_PP" : "_TP");
                ModelParityTopologyDefinition topology{
                    .test_id = id,
                    .kind = pipeline ? ModelParityTopologyKind::RankLocalPipelineParallel
                                     : ModelParityTopologyKind::RankLocalTensorParallel,
                    .participants = {{cuda ? GlobalDeviceAddress::cuda(0) : GlobalDeviceAddress::rocm(0), 0},
                                     {cuda ? GlobalDeviceAddress::cuda(1) : GlobalDeviceAddress::rocm(1), 0}},
                    .collective = cuda ? Collective::NCCL : Collective::RCCL,
                    .mpi_ranks = 1,
                };
                if (pipeline) topology.pipeline_stage_sizes = {1, 1};
                definitions.push_back(qwen38DenseParityDefinition(std::move(topology),
                    "pytorch_qwen38_dense_" + id + "_snapshots"));
            }
        definitions.push_back(qwen38DenseParityDefinition({
            .test_id = "CUDA2_ROCm2_TPPP",
            .kind = ModelParityTopologyKind::RankLocalPipelineParallel,
            .participants = {{GlobalDeviceAddress::cuda(0), 0}, {GlobalDeviceAddress::cuda(1), 0},
                             {GlobalDeviceAddress::rocm(0), 0}, {GlobalDeviceAddress::rocm(1), 0}},
            .collective = Collective::HETEROGENEOUS,
            .mpi_ranks = 1,
            .pipeline_stage_sizes = {2, 2},
            .tensor_parallel_collective = Collective::HETEROGENEOUS,
            .pipeline_tensor_parallel_collectives = {Collective::NCCL, Collective::RCCL},
        }, "pytorch_qwen38_dense_cuda2_rocm2_tppp_snapshots"));
        return definitions;
    }
} // namespace llaminar2::test::parity::qwen38
