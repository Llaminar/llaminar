/**
 * @file PartialWeightLoader.cpp
 * @brief Implementation of IPartialWeightLoader for pipeline parallelism
 *
 * Part of Phase 3: Pipeline Parallelism Integration
 *
 * Enables memory-efficient partial weight loading where each PP stage only
 * loads the weights it needs.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "IPartialWeightLoader.h"
#include "WeightManager.h"
#include "ModelContext.h"
#include "../execution/mpi_orchestration/IExecutionPlanBuilder.h"
#include "../utils/Logger.h"
#include <sstream>

namespace llaminar2
{

    // =========================================================================
    // PartialWeightLoader Implementation
    // =========================================================================

    class PartialWeightLoader : public IPartialWeightLoader
    {
    public:
        explicit PartialWeightLoader(const ModelConfig *model_config)
            : model_config_(model_config)
        {
        }

        std::vector<std::string> loadWeightsForPlan(
            WeightManager &weight_manager,
            const RankExecutionPlan &plan,
            const std::string &model_path) override
        {
            (void)model_path; // Path is used by weight manager internally

            auto needed = weightsForLayerRange(
                plan.first_layer,
                plan.last_layer,
                plan.has_embedding,
                plan.has_lm_head);

            LOG_INFO("[PartialWeightLoader] Loading " << needed.size()
                                                      << " weights for rank " << plan.rank
                                                      << " (layers " << plan.first_layer << "-" << plan.last_layer
                                                      << ", embedding=" << (plan.has_embedding ? "yes" : "no")
                                                      << ", lm_head=" << (plan.has_lm_head ? "yes" : "no") << ")");

            std::vector<std::string> loaded;
            loaded.reserve(needed.size());

            for (const auto &name : needed)
            {
                // Extract layer index from name for placement map lookup
                int layer_idx = extractLayerIndex(name);

                // Load weight via weight manager (handles caching, sharding, device placement)
                auto weight = weight_manager.getWeightForDevice(name, DeviceId::cpu(), layer_idx);
                if (weight)
                {
                    loaded.push_back(name);
                    LOG_DEBUG("[PartialWeightLoader] Loaded: " << name);
                }
                else
                {
                    LOG_WARN("[PartialWeightLoader] Failed to load: " << name);
                }
            }

            LOG_INFO("[PartialWeightLoader] Successfully loaded " << loaded.size()
                                                                  << "/" << needed.size() << " weights");

            return loaded;
        }

        std::vector<std::string> weightsForLayerRange(
            int first_layer,
            int last_layer,
            bool include_embedding,
            bool include_lm_head) const override
        {
            std::vector<std::string> weights;

            // Estimate capacity: 9 weights per layer + embedding + lm_head
            int layer_count = (last_layer >= first_layer) ? (last_layer - first_layer + 1) : 0;
            weights.reserve(layer_count * 9 + 3); // 9 per layer + embd + norm + output

            // Token embedding (first PP stage only)
            if (include_embedding)
            {
                weights.push_back("token_embd.weight");
            }

            // Per-layer weights
            for (int layer = first_layer; layer <= last_layer; ++layer)
            {
                auto layer_weights = allWeightsForLayer(layer);
                weights.insert(weights.end(), layer_weights.begin(), layer_weights.end());
            }

            // Output norm and LM head (last PP stage only)
            if (include_lm_head)
            {
                weights.push_back("output_norm.weight");
                weights.push_back("output.weight"); // LM head
            }

            return weights;
        }

        PartialWeightInfo getWeightInfoForPlan(
            const RankExecutionPlan &plan) const override
        {
            PartialWeightInfo info;

            info.weight_names = weightsForLayerRange(
                plan.first_layer,
                plan.last_layer,
                plan.has_embedding,
                plan.has_lm_head);

            info.layer_count = plan.layerCount();
            info.has_embedding = plan.has_embedding;
            info.has_lm_head = plan.has_lm_head;

            // Estimate memory (rough approximation - no model path available in this overload)
            info.estimated_bytes = estimateMemoryForPlan(plan, "");

            return info;
        }

        size_t estimateMemoryForPlan(
            const RankExecutionPlan &plan,
            const std::string & /*model_path*/) const override
        {
            // If we have model config, use it for estimation
            if (model_config_ && model_config_->estimated_weight_bytes > 0)
            {
                int total_layers = model_config_->n_layers;
                int my_layers = plan.layerCount();

                if (total_layers <= 0)
                {
                    return 0;
                }

                // Base estimate: proportional to layer count
                double layer_fraction = static_cast<double>(my_layers) / total_layers;
                size_t layer_bytes = static_cast<size_t>(
                    model_config_->estimated_weight_bytes * layer_fraction);

                // Add embedding estimate (~vocab_size * hidden_size * 2 bytes for Q4)
                size_t embedding_bytes = 0;
                if (plan.has_embedding && model_config_->vocab_size > 0 && model_config_->hidden_size > 0)
                {
                    // Approximate Q4 quantized size: 0.5 bytes per element + scale overhead
                    embedding_bytes = static_cast<size_t>(
                        model_config_->vocab_size * model_config_->hidden_size * 0.55);
                }

                // Add LM head estimate (similar to embedding)
                size_t lm_head_bytes = 0;
                if (plan.has_lm_head && model_config_->vocab_size > 0 && model_config_->hidden_size > 0)
                {
                    lm_head_bytes = static_cast<size_t>(
                        model_config_->vocab_size * model_config_->hidden_size * 0.55);
                }

                return layer_bytes + embedding_bytes + lm_head_bytes;
            }

            // Fallback: rough estimate based on typical Qwen2 layer sizes
            // Qwen2-7B: ~140MB per layer, ~300MB embedding, ~300MB LM head
            constexpr size_t BYTES_PER_LAYER = 140ULL * 1024 * 1024;
            constexpr size_t EMBEDDING_BYTES = 300ULL * 1024 * 1024;
            constexpr size_t LM_HEAD_BYTES = 300ULL * 1024 * 1024;

            size_t estimate = static_cast<size_t>(plan.layerCount()) * BYTES_PER_LAYER;
            if (plan.has_embedding)
            {
                estimate += EMBEDDING_BYTES;
            }
            if (plan.has_lm_head)
            {
                estimate += LM_HEAD_BYTES;
            }

            return estimate;
        }

        std::vector<std::string> attentionWeightsForLayer(int layer_idx) const override
        {
            std::vector<std::string> weights;
            weights.reserve(5);

            std::string prefix = "blk." + std::to_string(layer_idx) + ".";

            // Attention weights
            weights.push_back(prefix + "attn_q.weight");
            weights.push_back(prefix + "attn_k.weight");
            weights.push_back(prefix + "attn_v.weight");
            weights.push_back(prefix + "attn_output.weight");
            weights.push_back(prefix + "attn_norm.weight");

            return weights;
        }

        std::vector<std::string> ffnWeightsForLayer(int layer_idx) const override
        {
            std::vector<std::string> weights;
            weights.reserve(4);

            std::string prefix = "blk." + std::to_string(layer_idx) + ".";

            // FFN weights
            weights.push_back(prefix + "ffn_gate.weight");
            weights.push_back(prefix + "ffn_up.weight");
            weights.push_back(prefix + "ffn_down.weight");
            weights.push_back(prefix + "ffn_norm.weight");

            return weights;
        }

        std::vector<std::string> allWeightsForLayer(int layer_idx) const override
        {
            std::vector<std::string> weights;
            weights.reserve(9);

            auto attn = attentionWeightsForLayer(layer_idx);
            weights.insert(weights.end(), attn.begin(), attn.end());

            auto ffn = ffnWeightsForLayer(layer_idx);
            weights.insert(weights.end(), ffn.begin(), ffn.end());

            return weights;
        }

    private:
        const ModelConfig *model_config_;

        /**
         * @brief Extract layer index from weight name
         *
         * Parses names like "blk.5.attn_q.weight" to extract layer index (5).
         * Returns -1 for non-layer weights like "token_embd.weight".
         */
        static int extractLayerIndex(const std::string &name)
        {
            // Look for "blk.X." pattern
            const std::string prefix = "blk.";
            auto pos = name.find(prefix);
            if (pos == std::string::npos)
            {
                return -1; // Not a layer weight
            }

            pos += prefix.length();
            auto end_pos = name.find('.', pos);
            if (end_pos == std::string::npos)
            {
                return -1;
            }

            try
            {
                return std::stoi(name.substr(pos, end_pos - pos));
            }
            catch (...)
            {
                return -1;
            }
        }
    };

    // =========================================================================
    // Factory Implementation
    // =========================================================================

    std::unique_ptr<IPartialWeightLoader> createPartialWeightLoader(
        const ModelConfig *model_config)
    {
        return std::make_unique<PartialWeightLoader>(model_config);
    }

} // namespace llaminar2
