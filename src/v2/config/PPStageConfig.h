/**
 * @file PPStageConfig.h
 * @brief Configuration for a Pipeline Parallel stage
 *
 * Part of the Unified PP Graph Architecture (Phase 1.2).
 * A PP stage is a contiguous range of transformer layers executed on a
 * specific TP domain. Stages are executed sequentially with activation
 * transfers between them.
 *
 * @see docs/v2/projects/2026-02/UNIFIED_PP_GRAPH_ARCHITECTURE_PLAN.md
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include <string>

namespace llaminar2
{

/**
 * @brief Configuration for a Pipeline Parallel stage
 *
 * A PP stage is a contiguous range of layers executed on a specific TP domain.
 * Stages are executed sequentially with activation transfers between them.
 *
 * ## Layer Range Semantics
 * - Layer range is [first_layer, last_layer) - first is inclusive, last is exclusive
 * - For a 24-layer model with 3 stages:
 *   - Stage 0: first_layer=0, last_layer=8 (executes layers 0-7)
 *   - Stage 1: first_layer=8, last_layer=16 (executes layers 8-15)
 *   - Stage 2: first_layer=16, last_layer=24 (executes layers 16-23)
 *
 * ## Embedding and LM Head Ownership
 * - Stage 0 (first stage) owns the token embedding lookup (has_embedding=true)
 * - Final stage owns output_norm and LM head projection (has_lm_head=true)
 * - Middle stages have both flags set to false
 *
 * ## Domain Binding
 * - Each stage references a TPDomainConfig by name via domain_name
 * - The domain determines which devices execute this stage's computations
 *
 * @code
 * // Example: 3-stage PP across heterogeneous TP domains
 * PPStageConfig stage0 = PPStageConfig::firstStage(0, "gpu_nvidia", 0, 8);
 * PPStageConfig stage1 = PPStageConfig::middleStage(1, "gpu_amd", 8, 16);
 * PPStageConfig stage2 = PPStageConfig::lastStage(2, "cpu_tp", 16, 24);
 * @endcode
 */
struct PPStageConfig
{
    /// Stage ID (0, 1, 2, ...)
    int stage_id = 0;

    /// Name of the TP domain this stage runs on (references TPDomainConfig::name)
    std::string domain_name;

    /// First layer index (inclusive)
    int first_layer = 0;

    /// Last layer index (exclusive)
    int last_layer = 0;

    /// True if this stage includes the embedding lookup
    bool has_embedding = false;

    /// True if this stage includes the LM head (final projection)
    bool has_lm_head = false;

    // =========================================================================
    // Accessors
    // =========================================================================

    /**
     * @brief Get the number of layers in this stage
     * @return Layer count (last_layer - first_layer)
     */
    [[nodiscard]] int numLayers() const;

    /**
     * @brief Check if a layer index is in this stage
     * @param layer_idx The layer index to check
     * @return true if layer_idx is in [first_layer, last_layer)
     */
    [[nodiscard]] bool containsLayer(int layer_idx) const;

    /**
     * @brief Check if this is the first stage (has embedding)
     * @return true if has_embedding is set
     */
    [[nodiscard]] bool isFirstStage() const;

    /**
     * @brief Check if this is the last stage (has LM head)
     * @return true if has_lm_head is set
     */
    [[nodiscard]] bool isLastStage() const;

    // =========================================================================
    // Validation
    // =========================================================================

    /**
     * @brief Validate the configuration
     *
     * Checks that:
     * - stage_id >= 0
     * - domain_name is not empty
     * - first_layer >= 0
     * - last_layer > first_layer (at least one layer)
     * - last_layer <= total_layers
     * - If has_embedding, first_layer must be 0
     * - If has_lm_head, last_layer must equal total_layers
     *
     * @param total_layers Total layers in the model (for range validation)
     * @param error_msg Output: error message if invalid (optional)
     * @return true if valid
     */
    [[nodiscard]] bool validate(int total_layers, std::string *error_msg = nullptr) const;

    // =========================================================================
    // Factory Methods
    // =========================================================================

    /**
     * @brief Create a stage for the full model (single-stage PP)
     *
     * Creates a stage that spans all layers, with both embedding and LM head.
     * Use this when there's no pipeline parallelism.
     *
     * @param num_layers Total number of transformer layers
     * @param domain_name Name of the TP domain to run on
     * @return PPStageConfig covering all layers
     */
    static PPStageConfig fullModel(int num_layers, const std::string &domain_name);

    /**
     * @brief Create the first stage (with embedding)
     *
     * Creates a stage starting at layer 0 with embedding lookup enabled.
     *
     * @param stage_id Stage identifier (typically 0)
     * @param domain_name Name of the TP domain to run on
     * @param first_layer First layer index (must be 0)
     * @param last_layer Last layer index (exclusive)
     * @return PPStageConfig for the first stage
     */
    static PPStageConfig firstStage(int stage_id, const std::string &domain_name,
                                    int first_layer, int last_layer);

    /**
     * @brief Create a middle stage (no embedding, no LM head)
     *
     * Creates a stage for intermediate layers without embedding or LM head.
     *
     * @param stage_id Stage identifier
     * @param domain_name Name of the TP domain to run on
     * @param first_layer First layer index (inclusive)
     * @param last_layer Last layer index (exclusive)
     * @return PPStageConfig for a middle stage
     */
    static PPStageConfig middleStage(int stage_id, const std::string &domain_name,
                                     int first_layer, int last_layer);

    /**
     * @brief Create the last stage (with LM head)
     *
     * Creates a stage ending at the final layer with LM head enabled.
     *
     * @param stage_id Stage identifier
     * @param domain_name Name of the TP domain to run on
     * @param first_layer First layer index (inclusive)
     * @param last_layer Last layer index (exclusive, should equal total_layers)
     * @return PPStageConfig for the last stage
     */
    static PPStageConfig lastStage(int stage_id, const std::string &domain_name,
                                   int first_layer, int last_layer);
};

} // namespace llaminar2
