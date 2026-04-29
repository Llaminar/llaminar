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

        std::string getStopThinkingPrompt() const override
        {
            return "Considering the limited time by the user, I have to give the "
                   "solution based on the thinking directly now.\n</think>\n\n";
        }

        WeightShardingConfig getWeightShardingConfig() const override
        {
            // Start with dense Qwen3.5 sharding config (attention weights)
            Qwen35SchemaFactory dense_factory;
            auto config = dense_factory.getWeightShardingConfig();

            // MoE expert weights are replicated (each device processes all local experts)
            config.patterns.push_back(
                {"ffn_gate_exps.weight", WeightShardingMode::Replicate, WeightDimensionType::FFNHidden,
                 "MoE expert gate weights - replicated"});
            config.patterns.push_back(
                {"ffn_up_exps.weight", WeightShardingMode::Replicate, WeightDimensionType::FFNHidden,
                 "MoE expert up weights - replicated"});
            config.patterns.push_back(
                {"ffn_down_exps.weight", WeightShardingMode::Replicate, WeightDimensionType::FFNHidden,
                 "MoE expert down weights - replicated"});
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
                {"moe_expert_indices", {"seq_len", "moe_top_k"}, "fp32", BufferSemantic::Scratch,
                 "moe_scratch", 10, "MoE top-k expert indices per token (as float)"});
            schema.layer_buffers.push_back(
                {"moe_expert_weights", {"seq_len", "moe_top_k"}, "fp32", BufferSemantic::Scratch,
                 "moe_scratch", 9, "MoE top-k routing weights per token"});

            // Expert compute output: combined weighted expert output
            schema.layer_buffers.push_back(
                {"moe_combined_output", {"seq_len", "d_model"}, "fp32", BufferSemantic::Scratch,
                 "moe_output_scratch", 10, "Combined routed expert FFN output"});

            // Shared expert output
            schema.layer_buffers.push_back(
                {"moe_shared_expert_output", {"seq_len", "d_model"}, "fp32", BufferSemantic::Scratch,
                 "moe_output_scratch", 5, "Shared expert FFN output"});

            // Expert GEMM scratch buffers (for gate/up projections)
            schema.layer_buffers.push_back(
                {"moe_gate_scratch", {"seq_len", "moe_expert_intermediate"}, "fp32", BufferSemantic::Scratch,
                 "moe_gemm_scratch", 10, "Expert gate projection scratch"});
            schema.layer_buffers.push_back(
                {"moe_up_scratch", {"seq_len", "moe_expert_intermediate"}, "fp32", BufferSemantic::Scratch,
                 "moe_gemm_scratch", 5, "Expert up projection scratch"});

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
            config["MOE_EXPERT_FFN"] = SnapshotShardingMode::REPLICATED;
            config["MOE_SHARED_EXPERT"] = SnapshotShardingMode::REPLICATED;

            return config;
        }
    };

} // namespace llaminar2
