/**
 * @file DeepSeekV4GraphConfigBuilder.h
 * @brief Declarative model geometry and typed weight wiring for DeepSeek-V4.
 *
 * The authenticated GGUF contract (DeepSeekV4Config) owns every dimension and
 * per-layer policy. Shared orchestration still owns placement and graph
 * lifecycle. Every main layer is attention (no GDN); routed/shared experts keep
 * their ordinary MoE roles, while hyper-connection, compressor, indexer and
 * routing-policy tensors bind through DeepSeekV4LayerWeights.
 */
#pragma once
#include "../qwen/QwenStandardGraphConfigBuilder.h"

namespace llaminar2
{
struct GGUFModel;

/** @brief Metadata-only specialization; it never selects an execution backend. */
class DeepSeekV4GraphConfigBuilder final : public QwenStandardGraphConfigBuilder
{
public:
    /** @brief Authenticate exact metadata and populate the participant's model contract. */
    bool populateFromModelContext(IModelContext &ctx, GraphConfig &config) override;
    /** @brief Apply an exact directory to an existing local layer range. */
    static void populateFromDirectory(const GGUFModel &model, GraphConfig &config);
    /** @brief Bind ordinary MoE roles plus the DeepSeek-V4 extension roles. */
    ModelWeights buildWeights(WeightAccessor get_weight) override;
};
} // namespace llaminar2
