/**
 * @file EagerWeightValidator.h
 * @brief Schema-aware weight validation for model loading
 *
 * Validates that a model contains all required weights as declared by its
 * architecture schema. Optional weights (e.g., QKV biases in Qwen3) are
 * silently skipped; missing required weights produce a clear fatal error.
 *
 * @author David Sanftenberg
 * @date March 2026
 */

#pragma once

#include "../local_execution/graph/GraphSchema.h"
#include <string>
#include <vector>
#include <functional>

namespace llaminar2
{

    /**
     * @brief Result of eager weight validation
     */
    struct WeightValidationResult
    {
        bool success = true;

        /// Required weights that are missing from the model (fatal)
        std::vector<std::string> missing_required;

        /// Optional weights that are missing from the model (informational)
        std::vector<std::string> missing_optional;

        /// Weights that will be loaded (present in model, or optional+present)
        std::vector<std::pair<std::string, bool>> weights_to_load; // {name, is_optional}

        /// Human-readable error message (empty on success)
        std::string error_message() const
        {
            if (success)
                return {};
            std::string msg = "Missing required weight(s): ";
            for (size_t i = 0; i < missing_required.size(); ++i)
            {
                if (i > 0)
                    msg += ", ";
                msg += missing_required[i];
            }
            return msg;
        }
    };

    /**
     * @brief Validate model weights against architecture schema
     *
     * For each layer, checks all possible weights declared in the weight list.
     * Uses the schema factory to determine required vs optional weights, and
     * queries the model to check which tensors exist.
     *
     * Required weights that are missing cause validation failure.
     * Optional weights that are missing are silently skipped.
     *
     * @param schema_factory Architecture schema (determines required vs optional)
     * @param n_layers Number of transformer layers
     * @param has_tensor Predicate that checks if a tensor name exists in the model
     * @param first_layer First layer index (for PP stage validation, default 0)
     * @param last_layer Last layer index exclusive (for PP, default -1 = n_layers)
     * @return Validation result with lists of missing weights and loadable weights
     */
    inline WeightValidationResult validateLayerWeights(
        const ISchemaFactory &schema_factory,
        int n_layers,
        std::function<bool(const std::string &)> has_tensor,
        int first_layer = 0,
        int last_layer = -1)
    {
        WeightValidationResult result;
        if (last_layer < 0)
            last_layer = n_layers;

        result.weights_to_load.reserve(static_cast<size_t>(last_layer - first_layer) * 15);

        // Get per-layer weight suffixes from the architecture schema
        const auto suffixes = schema_factory.layerWeightSuffixes();

        for (int layer_idx = first_layer; layer_idx < last_layer; ++layer_idx)
        {
            std::string prefix = "blk." + std::to_string(layer_idx) + ".";

            for (const auto &suffix : suffixes)
            {
                std::string weight_name = prefix + suffix;
                bool is_optional = schema_factory.isWeightOptional(weight_name);
                bool exists = has_tensor(weight_name);

                if (!exists)
                {
                    if (is_optional)
                    {
                        // Optional weight missing — skip silently
                        result.missing_optional.push_back(weight_name);
                    }
                    else
                    {
                        // Required weight missing — fatal
                        result.missing_required.push_back(weight_name);
                        result.success = false;
                    }
                }
                else
                {
                    // Weight exists — include in load list
                    result.weights_to_load.emplace_back(weight_name, is_optional);
                }
            }
        }

        return result;
    }

} // namespace llaminar2
