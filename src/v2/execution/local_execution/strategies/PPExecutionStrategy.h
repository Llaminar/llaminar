/**
 * @file PPExecutionStrategy.h
 * @brief Pipeline Parallelism execution strategy
 *
 * Part of the Unified Multi-Device Orchestration Architecture (Phase 3).
 * Implements sequential forward execution with activation transfer between stages.
 *
 * @see docs/v2/projects/2026-02/UNIFIED_MULTI_DEVICE_ORCHESTRATION_DESIGN.md
 * @see IExecutionStrategy.h
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "IExecutionStrategy.h"
#include "../../../collective/ILocalPPContext.h"
#include <memory>
#include <string>

namespace llaminar2
{

    /**
     * @brief Pipeline Parallelism execution strategy
     *
     * Implements sequential forward execution through PP stages with explicit
     * activation transfer between stages. Each stage processes a contiguous
     * range of layers:
     *
     * ```
     * Stage 0 (layers 0-11)    Stage 1 (layers 12-23)
     *         │                        │
     *    embedding                     │
     *    layers 0-11                layers 12-23
     *         │                    lm_head
     *         └─── transfer ───────────┘
     * ```
     *
     * ## Execution Flow
     *
     * 1. Stage 0: `runner->forward(tokens, seq_len)` (embeds tokens)
     * 2. Transfer: `pp_ctx->transfer(hidden, 0, 1)` (cuda:0 → cuda:1)
     * 3. Stage 1: `runner->forwardFromHidden(seq_len)` (continues from hidden)
     * 4. ... repeat for more stages ...
     * 5. Logits: Copy from last stage
     *
     * ## State Management
     *
     * Each stage's DeviceGraphOrchestrator has its own InferenceState:
     * - Own hidden buffer (receives transferred data)
     * - Own KV cache (for its layer range only)
     * - Own position tracking (all stages in sync)
     *
     * ## Thread Safety
     *
     * This strategy executes stages **sequentially** (no internal parallelism).
     * Thread-safe for concurrent access from orchestrator.
     */
    class PPExecutionStrategy : public IExecutionStrategy
    {
    public:
        /**
         * @brief Construct PP execution strategy
         *
         * @param pp_ctx LOCAL PP context for activation transfers (required)
         * @throws std::invalid_argument if pp_ctx is null
         */
        explicit PPExecutionStrategy(ILocalPPContext *pp_ctx);

        ~PPExecutionStrategy() override = default;

        // Non-copyable, movable
        PPExecutionStrategy(const PPExecutionStrategy &) = delete;
        PPExecutionStrategy &operator=(const PPExecutionStrategy &) = delete;
        PPExecutionStrategy(PPExecutionStrategy &&) noexcept = default;
        PPExecutionStrategy &operator=(PPExecutionStrategy &&) noexcept = default;

        // =====================================================================
        // IExecutionStrategy Interface
        // =====================================================================

        /**
         * @brief Execute sequential forward through PP stages
         *
         * For each stage in order:
         * 1. Execute stage's forward (or forwardFromHidden for non-first stages)
         * 2. If not last stage, transfer activations to next stage
         *
         * @param runners Device runners (one per PP stage, in stage order)
         * @param tokens Input tokens (used by stage 0 only)
         * @param seq_len Sequence length
         * @return true if all stages succeeded
         */
        bool executeForward(
            std::vector<DeviceGraphOrchestrator *> &runners,
            const int *tokens,
            int seq_len) override;

        /**
         * @brief Copy logits from last stage
         *
         * Logits are produced by the LM head, which is on the last PP stage.
         * This method copies them to the output buffer.
         *
         * @param runners Device runners
         * @param output_buffer Pre-allocated buffer [seq_len, vocab_size]
         * @param seq_len Sequence length
         * @return true on success
         */
        bool gatherLogits(
            std::vector<DeviceGraphOrchestrator *> &runners,
            TensorBase *output_buffer,
            size_t seq_len) override;

        /**
         * @brief Clear KV caches on all stages
         */
        void clearCaches(
            std::vector<DeviceGraphOrchestrator *> &runners) override;

        /**
         * @brief Get current position (same across all stages)
         */
        int getPosition(
            const std::vector<DeviceGraphOrchestrator *> &runners) const override;

        /**
         * @brief Get parallelism mode
         * @return ParallelismMode::PP_ONLY
         */
        ParallelismMode mode() const override { return ParallelismMode::PP_ONLY; }

        /**
         * @brief Get description
         */
        std::string description() const override;

        // =====================================================================
        // PP-Specific Methods
        // =====================================================================

        /**
         * @brief Get the LOCAL PP context
         */
        ILocalPPContext *ppContext() const { return pp_ctx_; }

        /**
         * @brief Get number of PP stages
         */
        int numStages() const { return pp_ctx_ ? pp_ctx_->numStages() : 0; }

    private:
        ILocalPPContext *pp_ctx_; ///< LOCAL PP context (non-owning)
    };

} // namespace llaminar2
