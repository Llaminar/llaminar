/**
 * @file TPPPValidator.h
 * @brief Model-aware validation for Tensor Parallelism and Pipeline Parallelism configurations
 *
 * Validates that TP/PP configurations are compatible with the loaded model's architecture.
 * This validation runs AFTER model loading but BEFORE weight sharding begins.
 *
 * Validation Categories:
 * 1. Head Divisibility - KV heads and Q heads must divide evenly by TP degree
 * 2. Dimension Divisibility - FFN, embedding, vocab must divide evenly
 * 3. PP Layer Assignment - Layers must divide evenly across PP stages
 * 4. Domain Configuration - Named domains must have valid head assignments
 * 5. Proportional TP - Weighted splits must result in integer head counts
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "OrchestrationConfig.h"
#include "../interfaces/IModelContext.h"
#include <vector>
#include <string>
#include <sstream>
#include <cmath>

namespace llaminar2
{

    /**
     * @brief Result of TP/PP validation against model context
     */
    struct TPPPValidationResult
    {
        bool valid = true;
        std::vector<std::string> errors;
        std::vector<std::string> warnings;

        /**
         * @brief Add an error (sets valid = false)
         */
        void addError(const std::string &error)
        {
            valid = false;
            errors.push_back(error);
        }

        /**
         * @brief Add a warning (does not affect validity)
         */
        void addWarning(const std::string &warning)
        {
            warnings.push_back(warning);
        }

        /**
         * @brief Format all errors and warnings for logging
         */
        std::string toString() const
        {
            std::ostringstream oss;
            if (!errors.empty())
            {
                oss << "Errors:\n";
                for (const auto &e : errors)
                {
                    oss << "  - " << e << "\n";
                }
            }
            if (!warnings.empty())
            {
                oss << "Warnings:\n";
                for (const auto &w : warnings)
                {
                    oss << "  - " << w << "\n";
                }
            }
            return oss.str();
        }
    };

    /**
     * @brief Validator for TP/PP configurations against model architecture
     *
     * Performs comprehensive validation that the requested parallelism configuration
     * is compatible with the loaded model's architecture (head counts, dimensions, etc.).
     *
     * Usage:
     *   auto result = TPPPValidator::validate(config, model_ctx);
     *   if (!result.valid) {
     *       LOG_ERROR("TP/PP validation failed:\n" << result.toString());
     *       return false;
     *   }
     */
    class TPPPValidator
    {
    public:
        /**
         * @brief Validate TP/PP configuration against model context
         *
         * @param config Orchestration configuration
         * @param model Model context with architecture parameters
         * @return Validation result with errors and warnings
         */
        static TPPPValidationResult validate(
            const OrchestrationConfig &config,
            const IModelContext &model)
        {
            TPPPValidationResult result;

            // =====================================================================
            // 1. CPU Layers Validation (always runs, even without TP/PP)
            // =====================================================================

            validateCPULayers(config, model, result);

            // Skip remaining validation if no parallelism configured
            if (config.tp_degree == 1 && config.pp_degree == 1)
            {
                return result;
            }

            // =====================================================================
            // 2. Tensor Parallelism Validation
            // =====================================================================

            if (config.tp_degree > 1)
            {
                validateTPHeadDivisibility(config, model, result);
                validateTPDimensionDivisibility(config, model, result);
                validateTPProportionalWeights(config, model, result);
            }

            // =====================================================================
            // 3. Pipeline Parallelism Validation
            // =====================================================================

            if (config.pp_degree > 1)
            {
                validatePPLayerAssignment(config, model, result);
            }

            // =====================================================================
            // 4. Named Domain Validation (Advanced)
            // =====================================================================

            if (config.usesNamedDomains())
            {
                validateNamedDomains(config, model, result);
            }

            // =====================================================================
            // 5. Combined TP + PP Validation
            // =====================================================================

            if (config.tp_degree > 1 && config.pp_degree > 1)
            {
                validateCombinedTPPP(config, model, result);
            }

            return result;
        }

    private:
        static int decoderLayerCountExcludingTrailingMTP(const IModelContext &model)
        {
            int n_layers = model.blockCount();
            if (n_layers <= 0)
            {
                return n_layers;
            }

            const std::string trailing_nextn_fc =
                "blk." + std::to_string(n_layers - 1) + ".nextn.eh_proj.weight";
            if (model.hasTensor(trailing_nextn_fc))
            {
                return n_layers - 1;
            }

            return n_layers;
        }

        // =========================================================================
        // TP Validation Helpers
        // =========================================================================

        /**
         * @brief Validate that head counts are divisible by TP degree
         *
         * Checks:
         * - n_kv_heads % tp_degree == 0 (for K/V projection sharding)
         * - n_heads % tp_degree == 0 (for Q projection sharding)
         */
        static void validateTPHeadDivisibility(
            const OrchestrationConfig &config,
            const IModelContext &model,
            TPPPValidationResult &result)
        {
            int n_heads = model.headCount();
            int n_kv_heads = model.headCountKV();
            int tp_degree = config.tp_degree;

            // KV heads divisibility (critical for K/V projection sharding)
            if (n_kv_heads % tp_degree != 0)
            {
                std::ostringstream oss;
                oss << "TP degree " << tp_degree << " is incompatible with model: "
                    << "n_kv_heads=" << n_kv_heads << " is not divisible by " << tp_degree << ". "
                    << "Valid TP degrees for this model: ";

                // Suggest valid TP degrees
                bool first = true;
                for (int d = 1; d <= n_kv_heads; ++d)
                {
                    if (n_kv_heads % d == 0)
                    {
                        if (!first)
                            oss << ", ";
                        oss << d;
                        first = false;
                    }
                }
                result.addError(oss.str());
            }

            // Query heads divisibility (critical for Q projection sharding)
            if (n_heads % tp_degree != 0)
            {
                std::ostringstream oss;
                oss << "TP degree " << tp_degree << " is incompatible with model: "
                    << "n_heads=" << n_heads << " is not divisible by " << tp_degree << ". "
                    << "Valid TP degrees for this model: ";

                bool first = true;
                for (int d = 1; d <= n_heads; ++d)
                {
                    if (n_heads % d == 0)
                    {
                        if (!first)
                            oss << ", ";
                        oss << d;
                        first = false;
                    }
                }
                result.addError(oss.str());
            }

            // GQA ratio check (should always pass if model is valid)
            if (n_kv_heads > 0 && n_heads % n_kv_heads != 0)
            {
                result.addWarning("Model has non-standard GQA ratio: n_heads=" +
                                  std::to_string(n_heads) + " not divisible by n_kv_heads=" +
                                  std::to_string(n_kv_heads));
            }
        }

        /**
         * @brief Validate that key dimensions are divisible by TP degree
         *
         * Checks:
         * - ffn_hidden % tp_degree == 0 (for Gate/Up projection sharding)
         * - vocab_size % tp_degree == 0 (for LM head sharding + AllGather logits)
         * - embedding_dim % tp_degree == 0 (soft check, only warning)
         */
        static void validateTPDimensionDivisibility(
            const OrchestrationConfig &config,
            const IModelContext &model,
            TPPPValidationResult &result)
        {
            int tp_degree = config.tp_degree;
            int ffn_hidden = model.feedForwardLength();
            int vocab_size = model.vocabSize();
            int embedding_dim = model.embeddingLength();

            // FFN hidden dimension (critical for Gate/Up sharding)
            if (ffn_hidden % tp_degree != 0)
            {
                std::ostringstream oss;
                oss << "TP degree " << tp_degree << " is incompatible with model: "
                    << "ffn_hidden=" << ffn_hidden << " is not divisible by " << tp_degree;
                result.addError(oss.str());
            }

            // Vocab size (critical for LM head column sharding)
            if (vocab_size % tp_degree != 0)
            {
                std::ostringstream oss;
                oss << "TP degree " << tp_degree << " is incompatible with model: "
                    << "vocab_size=" << vocab_size << " is not divisible by " << tp_degree << ". "
                    << "LM head output will require padding or will produce incorrect logit indices.";
                result.addError(oss.str());
            }

            // Embedding dimension (soft check - some sharding modes don't require this)
            if (embedding_dim % tp_degree != 0)
            {
                result.addWarning("embedding_dim=" + std::to_string(embedding_dim) +
                                  " is not divisible by TP degree " + std::to_string(tp_degree) +
                                  ". This may affect certain embedding sharding modes.");
            }
        }

        /**
         * @brief Validate proportional TP weights result in integer head assignments
         *
         * When using --tp-weights, each device must get a whole number of heads.
         * E.g., with 28 heads and weights [0.73, 0.27], device 0 gets 20.44 heads (invalid!)
         */
        static void validateTPProportionalWeights(
            const OrchestrationConfig &config,
            const IModelContext &model,
            TPPPValidationResult &result)
        {
            if (config.tp_weights.empty())
            {
                return; // No proportional weights, skip
            }

            int n_heads = model.headCount();
            int n_kv_heads = model.headCountKV();

            // Check query head assignment
            int total_assigned_heads = 0;
            for (size_t i = 0; i < config.tp_weights.size(); ++i)
            {
                float weight = config.tp_weights[i];
                float heads_f = weight * n_heads;
                int heads_i = static_cast<int>(std::round(heads_f));

                // Check if fractional (more than 0.01 difference from nearest integer)
                if (std::abs(heads_f - heads_i) > 0.01f)
                {
                    std::ostringstream oss;
                    oss << "Proportional TP weight " << weight << " for device " << i
                        << " results in fractional head count: " << heads_f << " Q heads. "
                        << "Adjust weights to produce integer head counts.";
                    result.addError(oss.str());
                }

                // Check minimum heads per device
                if (heads_i < 1)
                {
                    std::ostringstream oss;
                    oss << "Proportional TP weight " << weight << " for device " << i
                        << " results in " << heads_i << " Q heads. Each device must have >= 1 head.";
                    result.addError(oss.str());
                }

                total_assigned_heads += heads_i;
            }

            // Verify total matches
            if (total_assigned_heads != n_heads)
            {
                result.addWarning("Proportional head assignment totals " +
                                  std::to_string(total_assigned_heads) + " Q heads, model has " +
                                  std::to_string(n_heads) + ". Rounding may cause imbalance.");
            }

            // Same check for KV heads
            int total_assigned_kv = 0;
            for (size_t i = 0; i < config.tp_weights.size(); ++i)
            {
                float weight = config.tp_weights[i];
                float kv_heads_f = weight * n_kv_heads;
                int kv_heads_i = static_cast<int>(std::round(kv_heads_f));

                if (std::abs(kv_heads_f - kv_heads_i) > 0.01f)
                {
                    std::ostringstream oss;
                    oss << "Proportional TP weight " << weight << " for device " << i
                        << " results in fractional KV head count: " << kv_heads_f << " KV heads. "
                        << "Adjust weights to produce integer head counts.";
                    result.addError(oss.str());
                }

                if (kv_heads_i < 1)
                {
                    std::ostringstream oss;
                    oss << "Proportional TP weight " << weight << " for device " << i
                        << " results in " << kv_heads_i << " KV heads. Each device must have >= 1 KV head.";
                    result.addError(oss.str());
                }

                total_assigned_kv += kv_heads_i;
            }

            if (total_assigned_kv != n_kv_heads)
            {
                result.addWarning("Proportional KV head assignment totals " +
                                  std::to_string(total_assigned_kv) + " KV heads, model has " +
                                  std::to_string(n_kv_heads) + ". Rounding may cause imbalance.");
            }
        }

        // =========================================================================
        // PP Validation Helpers
        // =========================================================================

        /**
         * @brief Validate PP layer assignment
         *
         * Checks:
         * - n_layers % pp_degree == 0 (for equal split)
         * - PP stage definitions cover all layers
         * - No overlapping or missing layers
         */
        static void validatePPLayerAssignment(
            const OrchestrationConfig &config,
            const IModelContext &model,
            TPPPValidationResult &result)
        {
            int n_layers = decoderLayerCountExcludingTrailingMTP(model);
            int pp_degree = config.pp_degree;

            // For EQUAL split mode, layers must be evenly divisible
            if (config.pp_split == PPSplitMode::EQUAL)
            {
                if (n_layers % pp_degree != 0)
                {
                    std::ostringstream oss;
                    oss << "PP degree " << pp_degree << " with EQUAL split is incompatible with model: "
                        << "n_layers=" << n_layers << " is not divisible by " << pp_degree << ". "
                        << "Use --pp-split weighted or manual for uneven distribution.";
                    result.addError(oss.str());
                }
            }

            // For MANUAL split mode, validate stage definitions
            if (config.pp_split == PPSplitMode::MANUAL && !config.pp_stage_definitions.empty())
            {
                std::vector<bool> layer_covered(n_layers, false);

                for (const auto &stage : config.pp_stage_definitions)
                {
                    // Check layer range validity
                    if (stage.first_layer < 0 || stage.last_layer >= n_layers)
                    {
                        std::ostringstream oss;
                        oss << "PP stage " << stage.stage_id << " has invalid layer range ["
                            << stage.first_layer << "-" << stage.last_layer << "], model has "
                            << n_layers << " layers (0-" << (n_layers - 1) << ")";
                        result.addError(oss.str());
                        continue;
                    }

                    // Check for overlaps
                    for (int l = stage.first_layer; l <= stage.last_layer; ++l)
                    {
                        if (layer_covered[l])
                        {
                            std::ostringstream oss;
                            oss << "PP stage " << stage.stage_id << " overlaps: layer "
                                << l << " is already assigned to another stage";
                            result.addError(oss.str());
                        }
                        layer_covered[l] = true;
                    }
                }

                // Check for missing layers
                for (int l = 0; l < n_layers; ++l)
                {
                    if (!layer_covered[l])
                    {
                        result.addError("PP stage definitions do not cover layer " +
                                        std::to_string(l));
                    }
                }
            }
        }

        // =========================================================================
        // CPU Layers Validation
        // =========================================================================

        /**
         * @brief Validate CPU layer assignment (independent of PP mode)
         *
         * This runs even without PP enabled since --cpu-layers can be used alone.
         */
        static void validateCPULayers(
            const OrchestrationConfig &config,
            const IModelContext &model,
            TPPPValidationResult &result)
        {
            int n_layers = decoderLayerCountExcludingTrailingMTP(model);

            // Validate CPU layers don't exceed total
            if (config.cpu_layers > n_layers)
            {
                std::ostringstream oss;
                oss << "--cpu-layers " << config.cpu_layers << " exceeds model layer count "
                    << n_layers;
                result.addError(oss.str());
            }
        }

        // =========================================================================
        // Named Domain Validation
        // =========================================================================

        /**
         * @brief Validate named domain configurations
         *
         * For each domain with proportional weights, validates that head assignment
         * produces valid integer counts.
         */
        static void validateNamedDomains(
            const OrchestrationConfig &config,
            const IModelContext &model,
            TPPPValidationResult &result)
        {
            int n_heads = model.headCount();
            int n_kv_heads = model.headCountKV();

            for (const auto &domain : config.domain_definitions)
            {
                int domain_tp_degree = static_cast<int>(domain.devices.size());

                // Check head divisibility for this domain
                if (n_kv_heads % domain_tp_degree != 0)
                {
                    std::ostringstream oss;
                    oss << "Domain '" << domain.name << "' has " << domain_tp_degree
                        << " devices, but model n_kv_heads=" << n_kv_heads
                        << " is not divisible by " << domain_tp_degree;
                    result.addError(oss.str());
                }

                if (n_heads % domain_tp_degree != 0)
                {
                    std::ostringstream oss;
                    oss << "Domain '" << domain.name << "' has " << domain_tp_degree
                        << " devices, but model n_heads=" << n_heads
                        << " is not divisible by " << domain_tp_degree;
                    result.addError(oss.str());
                }

                // Check proportional weights if specified
                if (domain.hasWeights())
                {
                    for (size_t i = 0; i < domain.weights.size(); ++i)
                    {
                        float weight = domain.weights[i];
                        float heads_f = weight * n_heads;
                        int heads_i = static_cast<int>(std::round(heads_f));

                        if (std::abs(heads_f - heads_i) > 0.01f || heads_i < 1)
                        {
                            std::ostringstream oss;
                            oss << "Domain '" << domain.name << "' device " << i
                                << " weight " << weight << " results in " << heads_f
                                << " heads (must be integer >= 1)";
                            result.addError(oss.str());
                        }
                    }
                }
            }
        }

        // =========================================================================
        // Combined TP + PP Validation
        // =========================================================================

        /**
         * @brief Validate combined TP + PP configuration
         *
         * Checks:
         * - Total parallelism doesn't exceed available resources
         * - TP and PP degrees are compatible
         */
        static void validateCombinedTPPP(
            const OrchestrationConfig &config,
            const IModelContext &model,
            TPPPValidationResult &result)
        {
            // With both TP and PP, each PP stage needs TP degree devices
            // Total devices needed = tp_degree * pp_degree
            int total_devices_needed = config.tp_degree * config.pp_degree;

            // Add informational message about combined parallelism
            if (total_devices_needed > 8)
            {
                result.addWarning("Combined TP+PP requires " +
                                  std::to_string(total_devices_needed) + " devices total (" +
                                  std::to_string(config.tp_degree) + " TP × " +
                                  std::to_string(config.pp_degree) + " PP)");
            }

            // Check that layers per PP stage is reasonable
            int n_layers = decoderLayerCountExcludingTrailingMTP(model);
            int layers_per_stage = n_layers / config.pp_degree;
            if (layers_per_stage < 1)
            {
                result.addError("PP degree " + std::to_string(config.pp_degree) +
                                " exceeds layer count " + std::to_string(n_layers) +
                                " - cannot have less than 1 layer per stage");
            }
        }
    };

} // namespace llaminar2
