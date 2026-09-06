/**
 * @file Qwen35MoESchema.h
 * @brief Declarative schema definition for Qwen3.5 MoE architecture
 *
 * Qwen3.5 MoE extends Qwen3.5 Dense with Mixture-of-Experts FFN:
 *   - Attention: identical to dense Qwen3.5 (GDN + FA hybrid)
 *   - FFN: SparseMoeBlock (router → top-K experts + shared expert + sigmoid gate)
 *
 * This schema factory reuses the Qwen3.5 attention templates and adds
 * MoE-specific FFN stages and weight sharding annotations.
 */

#pragma once

#include "../../execution/local_execution/graph/GraphSchema.h"
#include "../qwen35/Qwen35Schema.h"
#include "../qwen/QwenThinkingPolicy.h"
#include <string>

namespace llaminar2
{

    /**
     * @brief Schema factory for Qwen3.5 MoE architecture
     *
     * Extends Qwen35SchemaFactory with MoE-specific:
     * - Expert weight sharding (replicated per expert)
     * - Router stage specs
     * - MoE FFN template replacing dense SwiGLU
     */
    class Qwen35MoESchemaFactory : public ISchemaFactory
    {
    public:
        Qwen35MoESchemaFactory() = default;

        std::string architectureName() const override { return "qwen35moe"; }

        SamplingParams getRecommendedSamplingParams() const override
        {
            // Same as dense Qwen3.5
            SamplingParams params;
            params.temperature = 0.6f;
            params.top_p = 0.95f;
            params.top_k = 20;
            params.presence_penalty = 1.5f;
            return params;
        }

        /** @brief Return the shared Qwen continuation with its paragraph boundary. */
        std::string getStopThinkingPrompt() const override
        {
            return qwenStopThinkingPrompt();
        }

        WeightShardingConfig getWeightShardingConfig() const override
        {
            // Start with dense Qwen3.5 sharding config (attention weights)
            Qwen35SchemaFactory dense_factory;
            auto config = dense_factory.getWeightShardingConfig();

            // Routed MoE expert weights default to expert-id parallelism in TP.
            // InferenceRunnerFactory can override this to Replicate for explicit
            // --moe-routed-expert-compute replicated compatibility/debug runs.
            config.patterns.push_back(
                {"ffn_gate_exps.weight", WeightShardingMode::ExpertIdApportioned, WeightDimensionType::FFNHidden,
                 "MoE expert gate weights - expert-id parallel"});
            config.patterns.push_back(
                {"ffn_up_exps.weight", WeightShardingMode::ExpertIdApportioned, WeightDimensionType::FFNHidden,
                 "MoE expert up weights - expert-id parallel"});
            config.patterns.push_back(
                {"ffn_down_exps.weight", WeightShardingMode::ExpertIdApportioned, WeightDimensionType::FFNHidden,
                 "MoE expert down weights - expert-id parallel"});
            config.patterns.push_back(
                {"ffn_gate_inp.weight", WeightShardingMode::Replicate, WeightDimensionType::FFNHidden,
                 "MoE router weights - replicated"});
            config.patterns.push_back(
                {"ffn_gate_inp_shexp.weight", WeightShardingMode::Replicate, WeightDimensionType::FFNHidden,
                 "MoE shared expert gate - replicated"});

            // Shared expert weights
            config.patterns.push_back(
                {"ffn_gate_shexp.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::FFNHidden,
                 "Shared expert gate - column parallel"});
            config.patterns.push_back(
                {"ffn_up_shexp.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::FFNHidden,
                 "Shared expert up - column parallel"});
            config.patterns.push_back(
                {"ffn_down_shexp.weight", WeightShardingMode::InputParallel, WeightDimensionType::FFNHidden,
                 "Shared expert down - input parallel"});

            return config;
        }

        GraphSchema createSchema() const override
        {
            // Start with dense Qwen3.5 schema (has GDN + FA attention templates)
            Qwen35SchemaFactory dense_factory;
            GraphSchema schema = dense_factory.createSchema();

            // Override schema name
            schema.name = "qwen35moe";

            // Add MoE-specific activation buffers
            // Routing outputs: expert indices and weights for top-k selection
            schema.layer_buffers.push_back(
                {"moe_expert_indices", {"moe_activation_rows", "moe_top_k"}, "fp32", BufferSemantic::Scratch, "moe_scratch", 10, "MoE top-k expert indices per token (as float)"});
            schema.layer_buffers.push_back(
                {"moe_expert_weights", {"moe_activation_rows", "moe_top_k"}, "fp32", BufferSemantic::Scratch, "moe_scratch", 9, "MoE top-k routing weights per token"});

            // Expert compute output: combined weighted expert output
            schema.layer_buffers.push_back(
                {"moe_combined_output", {"moe_activation_rows", "d_model"}, "fp32", BufferSemantic::Scratch, "moe_output_scratch", 10, "Combined routed expert FFN output"});

            /*
             * LocalTP must retain every router slot until after the participant
             * collective. Canonical GPU publication extends that slot bank with
             * one shared-expert row bank per participant; the resolver leaves the
             * extra count at zero for topologies that publish branches
             * independently. This buffer deliberately has a separate lifetime
             * group because it coexists with the compact final outputs.
             */
            schema.layer_buffers.push_back(
                {"moe_canonical_route_contributions", {"moe_activation_rows", "moe_canonical_publication_slots", "d_model"}, "fp32", BufferSemantic::Scratch, "moe_canonical_route_scratch", 10, "Canonical route evidence: dense GPU slots/rank banks or packed indexed CPU rows plus a count trailer"});

            // Shared expert output
            schema.layer_buffers.push_back(
                {"moe_shared_expert_output", {"moe_activation_rows", "d_model"}, "fp32", BufferSemantic::Scratch, "moe_output_scratch", 5, "Shared expert FFN output"});

            // Expert GEMM scratch buffers (for gate/up projections)
            schema.layer_buffers.push_back(
                {"moe_gate_scratch", {"moe_activation_rows", "moe_ffn_intermediate_max"}, "fp32", BufferSemantic::Scratch, "moe_gemm_scratch", 10, "Routed/shared expert gate projection scratch"});
            schema.layer_buffers.push_back(
                {"moe_up_scratch", {"moe_activation_rows", "moe_ffn_intermediate_max"}, "fp32", BufferSemantic::Scratch, "moe_gemm_scratch", 5, "Routed/shared expert up projection scratch"});

            return schema;
        }

        bool isWeightOptional(const std::string &gguf_weight_name) const override
        {
            // Dense Qwen3.5 optional weights are also optional here
            Qwen35SchemaFactory dense_factory;
            if (dense_factory.isWeightOptional(gguf_weight_name))
                return true;

            // Dense FFN weights do NOT exist in MoE models (replaced by expert weights).
            // Mark them optional so the validator doesn't fail on missing ffn_gate/up/down.
            if (gguf_weight_name.find("ffn_gate.weight") != std::string::npos ||
                gguf_weight_name.find("ffn_up.weight") != std::string::npos ||
                gguf_weight_name.find("ffn_down.weight") != std::string::npos)
            {
                return true;
            }

            // MoE-specific weights are optional (not all layers are MoE in some configs)
            if (gguf_weight_name.find("ffn_gate_exps") != std::string::npos ||
                gguf_weight_name.find("ffn_up_exps") != std::string::npos ||
                gguf_weight_name.find("ffn_down_exps") != std::string::npos ||
                gguf_weight_name.find("ffn_gate_inp") != std::string::npos ||
                gguf_weight_name.find("ffn_gate_shexp") != std::string::npos ||
                gguf_weight_name.find("ffn_up_shexp") != std::string::npos ||
                gguf_weight_name.find("ffn_down_shexp") != std::string::npos ||
                gguf_weight_name.find("ffn_gate_inp_shexp") != std::string::npos)
            {
                return true;
            }

            return false;
        }

        std::vector<std::string> layerWeightSuffixes() const override
        {
            // Start with dense Qwen3.5 suffixes
            Qwen35SchemaFactory dense_factory;
            auto suffixes = dense_factory.layerWeightSuffixes();

            // Add MoE-specific weight suffixes
            suffixes.push_back("ffn_gate_exps.weight");
            suffixes.push_back("ffn_up_exps.weight");
            suffixes.push_back("ffn_down_exps.weight");
            suffixes.push_back("ffn_gate_inp.weight");
            suffixes.push_back("ffn_gate_inp_shexp.weight");
            suffixes.push_back("ffn_gate_shexp.weight");
            suffixes.push_back("ffn_up_shexp.weight");
            suffixes.push_back("ffn_down_shexp.weight");

            return suffixes;
        }

        StageShardingConfig getStageShardingConfig() const override
        {
            // Start with dense Qwen3.5 sharding config
            Qwen35SchemaFactory dense_factory;
            auto config = dense_factory.getStageShardingConfig();

            // Add MoE-specific stage sharding
            config["MOE_ROUTER"] = SnapshotShardingMode::REPLICATED;
            config["MOE_ROUTER_OUTPUT"] = SnapshotShardingMode::REPLICATED;
            config["MOE_ROUTING_INDICES"] = SnapshotShardingMode::REPLICATED;
            config["MOE_ROUTING_WEIGHTS"] = SnapshotShardingMode::REPLICATED;
            config["MOE_EXPERT_FFN"] = SnapshotShardingMode::REPLICATED;
            config["MOE_SHARED_EXPERT"] = SnapshotShardingMode::REPLICATED;
            config["MOE_EXPERT_OUTPUT"] = SnapshotShardingMode::ROW_PARALLEL;
            config["MOE_SHARED_EXPERT_OUTPUT"] = SnapshotShardingMode::ROW_PARALLEL;
            config["MOE_SHARED_GATE_OUTPUT"] = SnapshotShardingMode::ROW_PARALLEL;
            /*
             * MOE_COMBINED_OUTPUT is emitted only after every sharded routed
             * and shared-expert branch has completed its own collective.  The
             * final add therefore consumes replicated branch results and is
             * itself replicated.  Treating this boundary as row-parallel was
             * a stale description of the retired combined-output allreduce
             * graph and caused parity to request a snapshot from a node that
             * no longer exists.
             */
            config["MOE_COMBINED_OUTPUT"] = SnapshotShardingMode::REPLICATED;
            config["MOE_EXPERT_OUTPUT_ALLREDUCED"] = SnapshotShardingMode::REPLICATED;
            config["MOE_SHARED_EXPERT_OUTPUT_ALLREDUCED"] = SnapshotShardingMode::REPLICATED;

            /*
             * Before the publication collective, every original router slot
             * has exactly one participant owner and all non-owners publish
             * zero for that slot. Elementwise addition reconstructs the
             * complete canonical route tensor without changing its fixed
             * router-slot arithmetic order.
             */
            config["MOE_ROUTE_CONTRIBUTIONS"] =
                SnapshotShardingMode::ROW_PARALLEL;

            /*
             * Canonical rank-bank publication is an explicit three-edge TP
             * protocol. Each publish checkpoint contains disjoint routed and
             * shared banks, so it combines additively. The reduction exposes a
             * complete output on exactly its declared root. The subsequent
             * broadcast makes the finalized MoE row replicated again.
             * Packed CPU publication and the route-only rooted GPU lowering
             * share the same root-only snapshot ownership contract.
             */
            config["MOE_SHARED_RANK_BANK_PUBLISH"] =
                SnapshotShardingMode::ROW_PARALLEL;
            config["MOE_CANONICAL_PUBLICATION_REDUCE_TO_ROOT"] =
                SnapshotShardingMode::ROOT_ONLY;
            config["MOE_CANONICAL_ROUTES_GATHER_TO_ROOT"] =
                SnapshotShardingMode::ROOT_ONLY;
            config["MOE_CANONICAL_ROUTES_REDUCE_TO_ROOT"] =
                SnapshotShardingMode::ROOT_ONLY;
            /*
             * The continuation root publishes both typed views of the pinned
             * route epoch beside the rooted expert output: the domain-local
             * per-slot schedule and both global placement banks plus their
             * request selector. They are transaction evidence, not tensor
             * shards, and no participant may reconstruct them from setup-time
             * host topology.
             */
            config["MOE_DOMAIN_ROUTE_PARTICIPANT_IDS"] =
                SnapshotShardingMode::ROOT_ONLY;
            config["MOE_RUNTIME_ROUTE_WEIGHTS"] =
                SnapshotShardingMode::ROOT_ONLY;
            config["MOE_OVERLAY_ROUTE_PARTICIPANTS_BANK0"] =
                SnapshotShardingMode::ROOT_ONLY;
            config["MOE_OVERLAY_ROUTE_BANK0_EPOCH"] =
                SnapshotShardingMode::ROOT_ONLY;
            config["MOE_OVERLAY_ROUTE_PARTICIPANTS_BANK1"] =
                SnapshotShardingMode::ROOT_ONLY;
            config["MOE_OVERLAY_ROUTE_BANK1_EPOCH"] =
                SnapshotShardingMode::ROOT_ONLY;
            config["MOE_OVERLAY_ROUTE_SELECTED_BANK"] =
                SnapshotShardingMode::ROOT_ONLY;
            config["MOE_CANONICAL_PUBLICATION_BROADCAST"] =
                SnapshotShardingMode::REPLICATED;

            return config;
        }
    };

} // namespace llaminar2
