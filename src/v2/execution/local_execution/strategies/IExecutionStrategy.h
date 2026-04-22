/**
 * @file IExecutionStrategy.h
 * @brief Strategy interface for multi-device execution coordination
 *
 * Part of the Unified Multi-Device Orchestration Architecture (Phase 3).
 * Encapsulates the coordination logic that differs between parallelism modes:
 * - TP: parallel execution + allreduce
 * - PP: sequential execution + transfer
 * - TP+PP: hybrid (sequential stages, parallel within stage)
 *
 * @see docs/v2/UNIFIED_MULTI_DEVICE_ORCHESTRATION_DESIGN.md
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include <memory>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class DeviceGraphOrchestrator;
    class TensorBase;

    /**
     * @brief Parallelism mode for multi-device execution
     */
    enum class ParallelismMode
    {
        AUTO,          ///< Infer from configuration
        SINGLE_DEVICE, ///< One device, no parallelism
        TP_ONLY,       ///< Tensor parallelism only (horizontal sharding by heads)
        PP_ONLY,       ///< Pipeline parallelism only (vertical sharding by layers)
        TP_PLUS_PP,    ///< Both TP and PP (hybrid)
    };

    /**
     * @brief Convert ParallelismMode to string
     */
    inline const char *parallelismModeToString(ParallelismMode mode)
    {
        switch (mode)
        {
        case ParallelismMode::AUTO:
            return "AUTO";
        case ParallelismMode::SINGLE_DEVICE:
            return "SINGLE_DEVICE";
        case ParallelismMode::TP_ONLY:
            return "TP_ONLY";
        case ParallelismMode::PP_ONLY:
            return "PP_ONLY";
        case ParallelismMode::TP_PLUS_PP:
            return "TP_PLUS_PP";
        default:
            return "UNKNOWN";
        }
    }

    /**
     * @brief Strategy interface for multi-device execution coordination
     *
     * Defines the contract for coordinating execution across multiple devices.
     * Different parallelism modes implement this interface differently:
     *
     * - **TPExecutionStrategy**: Parallel forward on all devices, allreduce for
     *   row-parallel operations, allgather for column-parallel LM head.
     *
     * - **PPExecutionStrategy**: Sequential forward through stages, activation
     *   transfer between stages, logits from last stage only.
     *
     * - **TPPPExecutionStrategy**: Sequential through PP stages, parallel within
     *   each stage's TP domain, transfer between TP domains.
     *
     * ## Usage
     *
     * ```cpp
     * // In RankOrchestrator
     * std::unique_ptr<IExecutionStrategy> strategy_;
     *
     * bool forward(const int* tokens, int seq_len) {
     *     return strategy_->executeForward(device_runners_, tokens, seq_len);
     * }
     *
     * const float* logits() const {
     *     strategy_->gatherLogits(device_runners_, combined_logits_.get(), last_seq_len_);
     *     return combined_logits_->data();
     * }
     * ```
     *
     * ## Thread Safety
     *
     * Strategy implementations must be thread-safe for concurrent access from
     * the orchestrator. Internal parallelism (e.g., std::async in TP mode) is
     * handled by the strategy implementation.
     */
    class IExecutionStrategy
    {
    public:
        virtual ~IExecutionStrategy() = default;

        // =====================================================================
        // Core Execution
        // =====================================================================

        /**
         * @brief Execute forward pass across all devices
         *
         * Coordinates the forward pass according to the parallelism mode:
         * - TP: Launch all runners in parallel, wait for completion
         * - PP: Execute stages sequentially, transfer activations between
         * - TP+PP: Sequential stages, parallel within each stage
         *
         * @param runners Device runners to coordinate (one per device)
         * @param tokens Input token IDs (used by first/embedding stage)
         * @param seq_len Sequence length
         * @return true if forward pass succeeded on all devices/stages
         *
         * @note For PP modes, only the first stage uses `tokens`; later stages
         *       receive hidden state from previous stage via transfer.
         */
        virtual bool executeForward(
            std::vector<DeviceGraphOrchestrator *> &runners,
            const int *tokens,
            int seq_len) = 0;

        /**
         * @brief Gather final logits after forward pass
         *
         * Collects logits according to the parallelism mode:
         * - TP: AllGather partial logits from column-parallel LM head
         * - PP: Copy logits from last stage (which has the LM head)
         * - TP+PP: AllGather within last stage's TP domain
         *
         * @param runners Device runners
         * @param output_buffer Pre-allocated output buffer [seq_len, vocab_size]
         * @param seq_len Sequence length (for decode, typically 1)
         * @return true if gather succeeded
         *
         * @pre output_buffer must be pre-allocated with size [seq_len * vocab_size]
         * @post output_buffer contains combined logits from all devices/stages
         */
        virtual bool gatherLogits(
            std::vector<DeviceGraphOrchestrator *> &runners,
            TensorBase *output_buffer,
            size_t seq_len) = 0;

        // =====================================================================
        // State Management
        // =====================================================================

        /**
         * @brief Clear KV caches on all devices
         *
         * Resets KV cache state for a new generation. Called by orchestrator
         * when clear_cache() is invoked.
         *
         * @param runners Device runners to clear
         */
        virtual void clearCaches(
            std::vector<DeviceGraphOrchestrator *> &runners) = 0;

        /**
         * @brief Get current position in KV cache
         *
         * Returns the position across all devices:
         * - TP: All devices have same position (return any)
         * - PP: All stages have same position (return any)
         * - TP+PP: Same as above
         *
         * @param runners Device runners
         * @return Current position (0 if empty or no runners)
         */
        virtual int getPosition(
            const std::vector<DeviceGraphOrchestrator *> &runners) const = 0;

        // =====================================================================
        // Metadata
        // =====================================================================

        /**
         * @brief Get the parallelism mode this strategy implements
         *
         * @return ParallelismMode enum value
         */
        virtual ParallelismMode mode() const = 0;

        /**
         * @brief Get human-readable description of the strategy
         *
         * @return Description string (e.g., "2-stage PP with sequential execution")
         */
        virtual std::string description() const = 0;
    };

    // =========================================================================
    // Factory Functions (declared here, implemented in respective .cpp files)
    // =========================================================================

    class ILocalTPContext;
    class ILocalPPContext;

    /**
     * @brief Create TP execution strategy
     *
     * @param tp_ctx LOCAL TP context for collective operations
     * @return Unique pointer to TPExecutionStrategy
     */
    std::unique_ptr<IExecutionStrategy> createTPExecutionStrategy(
        ILocalTPContext *tp_ctx);

    /**
     * @brief Create PP execution strategy
     *
     * @param pp_ctx LOCAL PP context for activation transfers
     * @return Unique pointer to PPExecutionStrategy
     */
    std::unique_ptr<IExecutionStrategy> createPPExecutionStrategy(
        ILocalPPContext *pp_ctx);

    /**
     * @brief Create TP+PP hybrid execution strategy
     *
     * @param stage_tp_contexts TP contexts for each PP stage (one per stage)
     * @param pp_ctx LOCAL PP context for inter-stage transfers
     * @return Unique pointer to TPPPExecutionStrategy
     */
    std::unique_ptr<IExecutionStrategy> createTPPPExecutionStrategy(
        std::vector<std::unique_ptr<ILocalTPContext>> stage_tp_contexts,
        ILocalPPContext *pp_ctx);

} // namespace llaminar2
