/**
 * @file PPStageConfig.cpp
 * @brief Implementation of PPStageConfig
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "PPStageConfig.h"

namespace llaminar2
{

// =============================================================================
// Accessors
// =============================================================================

int PPStageConfig::numLayers() const
{
    return last_layer - first_layer;
}

bool PPStageConfig::containsLayer(int layer_idx) const
{
    return layer_idx >= first_layer && layer_idx < last_layer;
}

bool PPStageConfig::isFirstStage() const
{
    return has_embedding;
}

bool PPStageConfig::isLastStage() const
{
    return has_lm_head;
}

// =============================================================================
// Validation
// =============================================================================

bool PPStageConfig::validate(int total_layers, std::string *error_msg) const
{
    // Helper to set error message
    auto setError = [&](const std::string &msg) {
        if (error_msg)
        {
            *error_msg = msg;
        }
        return false;
    };

    // Check stage_id >= 0
    if (stage_id < 0)
    {
        return setError("stage_id must be >= 0, got " + std::to_string(stage_id));
    }

    // Check domain_name is not empty
    if (domain_name.empty())
    {
        return setError("domain_name must not be empty");
    }

    // Check first_layer >= 0
    if (first_layer < 0)
    {
        return setError("first_layer must be >= 0, got " + std::to_string(first_layer));
    }

    // Check last_layer > first_layer (at least one layer)
    if (last_layer <= first_layer)
    {
        return setError("last_layer (" + std::to_string(last_layer) +
                        ") must be > first_layer (" + std::to_string(first_layer) + ")");
    }

    // Check last_layer <= total_layers
    if (last_layer > total_layers)
    {
        return setError("last_layer (" + std::to_string(last_layer) +
                        ") must be <= total_layers (" + std::to_string(total_layers) + ")");
    }

    // Check embedding constraint: if has_embedding, first_layer must be 0
    if (has_embedding && first_layer != 0)
    {
        return setError("has_embedding is true but first_layer is " +
                        std::to_string(first_layer) + " (must be 0)");
    }

    // Check LM head constraint: if has_lm_head, last_layer must equal total_layers
    if (has_lm_head && last_layer != total_layers)
    {
        return setError("has_lm_head is true but last_layer (" + std::to_string(last_layer) +
                        ") != total_layers (" + std::to_string(total_layers) + ")");
    }

    // All checks passed
    if (error_msg)
    {
        error_msg->clear();
    }
    return true;
}

// =============================================================================
// Factory Methods
// =============================================================================

PPStageConfig PPStageConfig::fullModel(int num_layers, const std::string &domain_name)
{
    PPStageConfig config;
    config.stage_id = 0;
    config.domain_name = domain_name;
    config.first_layer = 0;
    config.last_layer = num_layers;
    config.has_embedding = true;
    config.has_lm_head = true;
    return config;
}

PPStageConfig PPStageConfig::firstStage(int stage_id, const std::string &domain_name,
                                        int first_layer, int last_layer)
{
    PPStageConfig config;
    config.stage_id = stage_id;
    config.domain_name = domain_name;
    config.first_layer = first_layer;
    config.last_layer = last_layer;
    config.has_embedding = true;
    config.has_lm_head = false;
    return config;
}

PPStageConfig PPStageConfig::middleStage(int stage_id, const std::string &domain_name,
                                         int first_layer, int last_layer)
{
    PPStageConfig config;
    config.stage_id = stage_id;
    config.domain_name = domain_name;
    config.first_layer = first_layer;
    config.last_layer = last_layer;
    config.has_embedding = false;
    config.has_lm_head = false;
    return config;
}

PPStageConfig PPStageConfig::lastStage(int stage_id, const std::string &domain_name,
                                       int first_layer, int last_layer)
{
    PPStageConfig config;
    config.stage_id = stage_id;
    config.domain_name = domain_name;
    config.first_layer = first_layer;
    config.last_layer = last_layer;
    config.has_embedding = false;
    config.has_lm_head = true;
    return config;
}

} // namespace llaminar2
