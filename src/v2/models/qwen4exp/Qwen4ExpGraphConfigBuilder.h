/**
 * @file Qwen4ExpGraphConfigBuilder.h
 * @brief Declarative model geometry and typed weight wiring for Flash Next.
 *
 * The exact GGUF contract owns all dimensions and layer identities. Shared
 * orchestration still owns placement, TP/PP and graph lifecycle. Branch mixers
 * replace ordinary residual normalization; PLE table payloads remain outside
 * model-weight materialization under the immutable row-storage authority.
 */
#pragma once
#include "../qwen/QwenStandardGraphConfigBuilder.h"

namespace llaminar2
{
struct GGUFModel;

/** @brief Metadata-only specialization; it never selects an execution backend. */
class Qwen4ExpGraphConfigBuilder final : public QwenStandardGraphConfigBuilder
{
public:
    /**
     * @brief Authenticate exact metadata and populate the participant's model contract.
     * @param ctx Loaded context, including its local PP layer range.
     * @param config Existing execution policy, preserved while model fields are set.
     * @return True after successful validation.
     * @throws std::invalid_argument If the exact GGUF directory is unavailable or invalid.
     */
    bool populateFromModelContext(IModelContext &ctx, GraphConfig &config) override;

    /**
     * @brief Apply an exact directory to an existing local layer range.
     * @param model Parsed split-merged immutable metadata and tensor inventory.
     * @param config Execution policy; zero n_layers selects the whole main decoder.
     * @throws std::invalid_argument On a malformed directory or out-of-range PP span.
     */
    static void populateFromDirectory(const GGUFModel &model, GraphConfig &config);

    /**
     * @brief Bind common GDN/MoE tensors and the model's branch/indexer/PLE extensions.
     * @param get_weight Exact prepared tensor accessor, with owner-retained lifetimes.
     * @return Model and per-layer role views; no ordinary final norm is substituted.
     */
    ModelWeights buildWeights(WeightAccessor get_weight) override;
};
} // namespace llaminar2
