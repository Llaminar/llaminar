/**
 * @file Qwen36ModelParityDefinitions.h
 * @brief Authenticated Qwen3.6 model identities and canonical parity topologies.
 *
 * Qwen3.6 reuses the Qwen3.5 hybrid GDN graph family and adds one recursive
 * next-token predictor. The predictor weights are reused at every logical MTP
 * depth, so the production and Hugging Face reference contracts admit the
 * complete depth-15 matrix without duplicating model tensors. This file owns
 * only declarative model/topology/economy values; production orchestration is
 * projected by ModelParityCase through the same server-facing configuration.
 */

#pragma once

#include "../ModelParityDefinition.h"
#include "../qwen35moe/Qwen35MoEModelParityDefinitions.h"
#include "Qwen36MTPCheckpointSurface.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2::test::parity::qwen36
{
    /** Exact short prompt shared by the canonical Qwen3.6 reference packs. */
    inline constexpr const char *kQwen36ParityPrompt =
        "The quick brown fox jumps over the lazy dog";

    /** @return Authenticated Qwen3.6 tokenization of the canonical prompt. */
    inline std::vector<int> qwen36ParityTokenIds()
    {
        return {760, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388};
    }

    /**
     * @brief Return the dense 27B Q4_K_S model/reference identity.
     *
     * @param reference_directory Backend-isolated authenticated reference pack.
     */
    inline ModelParityModelDefinition qwen36Dense27BQ4KSParityModel(
        std::string reference_directory =
            "pytorch_qwen36_dense_singledevice_snapshots")
    {
        return {
            .test_id = "Qwen36Dense_27B_Q4KS",
            .model_path = "/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf",
            .reference_directory = std::move(reference_directory),
            .prompt = kQwen36ParityPrompt,
            .token_ids = qwen36ParityTokenIds(),
            .decode_steps = 3,
            .max_seq_len = 4096,
            .transformer_layers = 64,
            .attention_heads = 24,
            .kv_heads = 4,
            .maximum_mtp_draft_depth =
                kModelParityRequiredMaximumMTPDepth,
            .mtp_checkpoint_surface =
                qwen36DenseMTPCheckpointSurface(),
        };
    }

    /**
     * @brief Return the 35B-A3B IQ3_S model/reference identity.
     *
     * @param reference_directory Backend-isolated authenticated reference pack.
     */
    inline ModelParityModelDefinition qwen36MoE35BIQ3SParityModel(
        std::string reference_directory)
    {
        return {
            .test_id = "Qwen36MoE_35B_IQ3S",
            .model_path =
                "/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf",
            .reference_directory = std::move(reference_directory),
            .prompt = kQwen36ParityPrompt,
            .token_ids = qwen36ParityTokenIds(),
            .decode_steps = 3,
            .max_seq_len = 4096,
            .transformer_layers = 40,
            .attention_heads = 16,
            .kv_heads = 2,
            .maximum_mtp_draft_depth =
                kModelParityRequiredMaximumMTPDepth,
        };
    }

    /** @return Strict dense Q4_K_S single-device numerical contract. */
    inline BackendThresholds qwen36DenseSingleDeviceThresholds()
    {
        return {
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.93f,
            .early_layers_count = 8,
            .min_early_layers_passed = 8,
            .kl_threshold = 0.08f,
            .min_top1_accuracy = 80.0f,
            .min_top5_accuracy = 80.0f,
            .pytorch_top1_in_topk = 3,
        };
    }

    /** @return Strict Qwen3.6 MoE single-device numerical contract. */
    inline BackendThresholds qwen36MoESingleDeviceThresholds()
    {
        return {
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.98f,
            .early_layers_count = 6,
            .min_early_layers_passed = 5,
            .kl_threshold = 0.03f,
            .min_top1_accuracy = 80.0f,
            .min_top5_accuracy = 60.0f,
            .pytorch_top1_in_topk = 3,
        };
    }

    /** @return Shard-aware Qwen3.6 MoE ExpertOverlay numerical contract. */
    inline BackendThresholds qwen36MoEExpertOverlayThresholds()
    {
        auto thresholds = qwen35moe::qwen35MoEMultiDeviceThresholds();
        thresholds.min_top5_accuracy = 60.0f;
        return thresholds;
    }

    /** @return One exact rank-local single-device topology. */
    inline ModelParityTopologyDefinition qwen36SingleDeviceTopology(
        std::string test_id,
        GlobalDeviceAddress device)
    {
        return {
            .test_id = std::move(test_id),
            .kind = ModelParityTopologyKind::SingleDevice,
            .participants = {{std::move(device), 0}},
            .collective = Collective::None,
            .mpi_ranks = 1,
        };
    }

    /**
     * @brief Return one homogeneous two-GPU ExpertOverlay topology.
     *
     * Capacity is deliberately automatic: the sole priority-zero fallback tier
     * fills to the production planner's exact admitted device capacity after
     * every named allocation. No model-specific expert-count or byte cap is
     * embedded in parity configuration.
     */
    inline ModelParityTopologyDefinition qwen36MoEGPU2ExpertOverlayTopology(
        std::string test_id,
        std::string domain_name,
        Collective collective,
        CollectiveBackendType backend,
        std::vector<GlobalDeviceAddress> participants)
    {
        ModelParityTopologyDefinition topology{
            .test_id = std::move(test_id),
            .kind = ModelParityTopologyKind::RankLocalTensorParallel,
            .collective = collective,
            .mpi_ranks = 1,
        };
        for (const auto &participant : participants)
            topology.participants.push_back({participant, 0});
        topology.expert_overlay_plan =
            qwen35moe::qwen35MoESingleDomainOverlayPlan(
                std::move(domain_name),
                ExecutionDomainScope::RANK_LOCAL,
                backend,
                std::move(participants));
        return topology;
    }

    /** @return Two-CUDA NCCL ExpertOverlay topology. */
    inline ModelParityTopologyDefinition qwen36MoECuda2ExpertOverlayTopology()
    {
        return qwen36MoEGPU2ExpertOverlayTopology(
            "LocalTP_NCCL_2xCUDA_ExpertOverlay",
            "qwen36_moe_cuda_local_tp",
            Collective::NCCL,
            CollectiveBackendType::NCCL,
            {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)});
    }

    /** @return Two-ROCm RCCL ExpertOverlay topology. */
    inline ModelParityTopologyDefinition qwen36MoERocm2ExpertOverlayTopology()
    {
        return qwen36MoEGPU2ExpertOverlayTopology(
            "LocalTP_RCCL_2xROCm_ExpertOverlay",
            "qwen36_moe_rocm_local_tp",
            Collective::RCCL,
            CollectiveBackendType::RCCL,
            {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
    }

    /**
     * @return One CPU ExpertOverlay tier spanning two sockets and two ranks.
     *
     * Dense tensor parallelism and routed expert ownership share the node
     * domain. Dynamic movement corrects participant skew within that tier;
     * there is no second priority and therefore no promotion/demotion axis.
     */
    inline ModelParityTopologyDefinition qwen36MoECPU2NodeTPTopology()
    {
        return {
            .test_id = "NodeTP_2xMPI_CPU_ExpertOverlay",
            .kind = ModelParityTopologyKind::NodeTensorParallel,
            .participants = {
                {GlobalDeviceAddress::cpu(0), 0},
                {GlobalDeviceAddress::cpu(1), 1},
            },
            .collective = Collective::MPI,
            .mpi_ranks = 2,
            .expert_overlay_plan = qwen35moe::qwen35MoESingleDomainOverlayPlan(
                "qwen36_moe_cpu_node_tp", ExecutionDomainScope::NODE_LOCAL,
                CollectiveBackendType::MPI,
                {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)}),
        };
    }

    /**
     * @brief Return short-request economics that force a real Dynamic epoch.
     *
     * The controller still evaluates the production payoff model and moves
     * prepared weights asynchronously. Only histogram/cadence geometry is
     * shortened so a five-step authenticated decode reaches the decision,
     * transfer, and apply boundaries.
     */
    inline MoERebalanceRuntimeConfig qwen36MoEDynamicParityEconomics()
    {
        auto config = qwen35moe::qwen35MoEDynamicParityEconomics();
        /*
         * The production policy expresses imbalance as a ratio in per-mille,
         * so 1000 is the neutral lower bound accepted by configuration
         * validation.  The remaining zero-improvement thresholds still make
         * the short parity trace admit an economical move without inventing
         * an invalid policy value solely for the test.
         */
        config.dynamic_imbalance_threshold_per_mille = 1000;
        /*
         * The first one-token boundary still forces physical movement.  Once
         * that epoch retires, leave one complete maximum-width MTP verifier
         * transaction between maintenance boundaries.  Otherwise a cadence
         * of one marks every adaptive-depth observation budget-limited and the
         * two independent device controllers can never both make progress.
         */
        config.device_min_maintenance_period_tokens =
            kModelParityRequiredMaximumMTPDepth + 1;
        config.release_raw_expert_weights = false;
        return config;
    }

    /** @return Canonical dense definition for one topology. */
    inline ModelParityDefinition qwen36DenseParityDefinition(
        ModelParityTopologyDefinition topology,
        std::string reference_directory)
    {
        ModelParityDefinition definition;
        definition.model = qwen36Dense27BQ4KSParityModel(
            std::move(reference_directory));
        definition.topology = std::move(topology);
        definition.thresholds = qwen36DenseSingleDeviceThresholds();
        definition.precisions.activation = {ActivationPrecision::FP32};
        definition.precisions.kv_cache = {KVCachePrecision::FP16};
        definition.features.mtp = ModelParityAxisProfile::Standard;
        return definition;
    }

    /** @return Canonical MoE definition for one topology. */
    inline ModelParityDefinition qwen36MoEParityDefinition(
        ModelParityTopologyDefinition topology,
        std::string reference_directory,
        BackendThresholds thresholds)
    {
        ModelParityDefinition definition;
        definition.model = qwen36MoE35BIQ3SParityModel(
            std::move(reference_directory));
        definition.topology = std::move(topology);
        definition.thresholds = std::move(thresholds);
        definition.precisions.activation = {ActivationPrecision::FP32};
        definition.precisions.kv_cache = {KVCachePrecision::FP16};
        definition.features.mtp = ModelParityAxisProfile::Standard;
        const auto dynamic_policy = qwen36MoEDynamicParityEconomics();
        definition.dynamic_rebalance = {
            .economic_movement = dynamic_policy,
            .economic_movement_and_observed_speedup = dynamic_policy,
        };
        definition.collective_evidence_source =
            ParityCollectiveEvidenceSource::PostCollectiveSnapshot;
        definition.tp_allreduce_precision_override = "schema";
        if (definition.topology.kind == ModelParityTopologyKind::SingleDevice &&
            definition.topology.participants.size() == 1 &&
            !definition.topology.participants.front().address.isCPU())
            definition.e2e_certifiable = {{.mtp = ModelParityMTP::DynamicDepth}};
        return definition;
    }
    /** @return Full CPU two-socket matrix with one Dynamic/adaptive HTTP tag. */
    inline ModelParityDefinition qwen36MoECPU2NodeTPParityDefinition()
    {
        auto definition = qwen36MoEParityDefinition(
            qwen36MoECPU2NodeTPTopology(),
            "pytorch_qwen36_moe_singledevice_cpu_snapshots",
            qwen36MoEExpertOverlayThresholds());
        definition.e2e_certifiable = {{
            .mtp = ModelParityMTP::DynamicDepth,
            .owner_order = RoutedExpertOwnerOrder::Ordinal,
            .movement = ModelParityExpertMovement::Dynamic,
        }};
        return definition;
    }
} // namespace llaminar2::test::parity::qwen36
