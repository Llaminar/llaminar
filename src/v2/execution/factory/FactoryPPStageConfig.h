/**
 * @file FactoryPPStageConfig.h
 * @brief PP stage configuration for inference runner factory
 *
 * This header is separate from InferenceRunnerFactory.h to avoid circular
 * dependencies with MultiDeviceOrchestrator.h
 */

#pragma once

namespace llaminar2
{

    /**
     * @brief Configuration for a single PP (Pipeline Parallel) stage
     *
     * Specifies which transformer layers belong to this stage and what
     * model components it owns. Used by InferenceRunnerFactory to create
     * PP stage runners, and by DeviceGraphOrchestrator to build partial graphs.
     *
     * Layer Ownership:
     * - First stage typically owns the token embedding lookup (has_embedding=true)
     * - Final stage typically owns output_norm and LM head projection (has_lm_head=true)
     * - Middle stages have both flags set to false
     */
    struct FactoryPPStageConfig
    {
        int first_layer = 0;        ///< First layer index this stage executes (inclusive)
        int last_layer = 0;         ///< Last layer index this stage executes (exclusive)
        bool has_embedding = false; ///< True if this stage owns the token embedding lookup
        bool has_lm_head = false;   ///< True if this stage owns output_norm and LM head projection

        /**
         * @brief Validate the PP stage configuration
         *
         * Checks that:
         * - first_layer >= 0 (valid layer index)
         * - last_layer > first_layer (at least one layer in range)
         * - Edge stages have appropriate ownership flags (first stage should have
         *   embedding OR last stage should have lm_head, but this is advisory)
         *
         * @return true if configuration is valid, false otherwise
         */
        [[nodiscard]] bool isValid() const
        {
            // Basic range validation
            if (first_layer < 0)
                return false;
            if (last_layer <= first_layer)
                return false;

            // Note: We don't enforce has_embedding/has_lm_head here since the
            // caller may be constructing a middle stage. The orchestrator
            // validates that exactly one stage has each flag across all stages.
            return true;
        }

        /**
         * @brief Get the number of layers this stage executes
         * @return Layer count (last_layer - first_layer)
         */
        [[nodiscard]] int layerCount() const
        {
            return last_layer - first_layer;
        }
    };

} // namespace llaminar2
