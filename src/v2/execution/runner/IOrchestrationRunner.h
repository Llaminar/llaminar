/**
 * @file IOrchestrationRunner.h
 * @brief Interface for orchestrated inference execution
 *
 * Provides the main interface for running inference with complex orchestration
 * scenarios including:
 * - Pipeline parallelism (PP) across ranks
 * - Local tensor parallelism (LOCAL TP) across devices within a rank
 * - Global tensor parallelism (GLOBAL TP) across ranks
 * - Heterogeneous device execution (CUDA, ROCm, CPU)
 *
 * This is the Phase 5 entry point that wires together all Phase 0-4 components:
 * - Phase 0: OrchestrationConfig from CLI/YAML
 * - Phase 1-2: RankExecutionPlan from ExecutionPlanBuilder
 * - Phase 3: Pipeline parallel graph building
 * - Phase 4: LOCAL TP context and weight sharding
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../../config/OrchestrationConfig.h"
#include "../mpi_orchestration/RankExecutionPlan.h"
#include "../../utils/Sampler.h"
#include <memory>
#include <string>
#include <vector>
#include <optional>

namespace llaminar2
{
    class ITokenizer;          // Forward declaration
    struct GraphExecutorStats; // Forward declaration
}

namespace llaminar2
{

    /**
     * @brief Result of a generation step or full generation
     */
    struct GenerationResult
    {
        std::vector<int32_t> tokens; ///< Generated tokens
        std::vector<float> logprobs; ///< Log probabilities (optional)
        bool is_complete{false};     ///< Whether generation is complete (EOS reached)
        std::string error;           ///< Error message if failed (empty on success)

        /**
         * @brief Check if generation was successful
         * @return true if no error occurred
         */
        bool success() const { return error.empty(); }

        /**
         * @brief Get number of tokens generated
         */
        size_t tokenCount() const { return tokens.size(); }
    };

    /**
     * @brief Interface for orchestrated inference
     *
     * This interface extends the basic IInferenceRunner concept to support
     * complex orchestration scenarios. It is mockable for unit testing.
     *
     * Lifecycle:
     * 1. Factory creates IOrchestrationRunner from config
     * 2. Call initialize() to load model, set up devices, build graphs
     * 3. Call generate() or prefill()/decodeStep() for inference
     * 4. Call shutdown() for clean teardown
     *
     * Thread safety: Not thread-safe. Each runner should be used from one thread.
     * For multi-threaded inference, create separate runners.
     */
    class IOrchestrationRunner
    {
    public:
        virtual ~IOrchestrationRunner() = default;

        // =====================================================================
        // Lifecycle
        // =====================================================================

        /**
         * @brief Initialize the runner
         *
         * Performs all initialization steps:
         * - Build execution plan from config
         * - Set up local TP context (if enabled)
         * - Load weights (with sharding)
         * - Build compute graphs
         * - Initialize device contexts
         *
         * @return true on success, false on failure (check lastError())
         */
        virtual bool initialize() = 0;

        /**
         * @brief Clean shutdown
         *
         * Releases all resources:
         * - Frees device memory
         * - Closes collective contexts
         * - Synchronizes pending operations
         *
         * Safe to call multiple times or if not initialized.
         */
        virtual void shutdown() = 0;

        // =====================================================================
        // Inference
        // =====================================================================

        /**
         * @brief Prefill phase: process input tokens
         *
         * Processes the prompt tokens and fills the KV cache.
         * After prefill, call decodeStep() to generate new tokens.
         *
         * @param tokens Input token IDs
         * @return true on success, false on failure (check lastError())
         */
        virtual bool prefill(const std::vector<int32_t> &tokens) = 0;

        /**
         * @brief Decode step: generate next token
         *
         * Generates a single token using the current KV cache state.
         * Must call prefill() first.
         *
         * Uses greedy sampling (argmax) unless sampling params were set.
         *
         * @return GenerationResult with the generated token
         */
        virtual GenerationResult decodeStep() = 0;

        /**
         * @brief Full generation loop
         *
         * Convenience method that runs prefill + decode loop.
         * Equivalent to:
         *   prefill(prompt_tokens);
         *   while (!complete && count < max_new_tokens) {
         *       result = decodeStep();
         *   }
         *
         * @param prompt_tokens Input prompt token IDs
         * @param max_new_tokens Maximum number of tokens to generate
         * @param sampling Sampling parameters (temperature, top_k, top_p)
         * @return GenerationResult with all generated tokens
         */
        virtual GenerationResult generate(
            const std::vector<int32_t> &prompt_tokens,
            int max_new_tokens,
            const SamplingParams &sampling) = 0;

        // =====================================================================
        // Configuration
        // =====================================================================

        /**
         * @brief Get the execution plan for this rank
         *
         * The execution plan describes what this rank should do:
         * - Which layers to execute
         * - Device assignments
         * - TP configuration
         * - PP neighbors
         *
         * @return Reference to the RankExecutionPlan
         */
        virtual const RankExecutionPlan &executionPlan() const = 0;

        /**
         * @brief Get orchestration configuration
         *
         * Returns the OrchestrationConfig used to create this runner.
         *
         * @return Reference to OrchestrationConfig
         */
        virtual const OrchestrationConfig &config() const = 0;

        // =====================================================================
        // Status
        // =====================================================================

        /**
         * @brief Check if the runner is initialized
         *
         * @return true if initialize() succeeded
         */
        virtual bool isInitialized() const = 0;

        /**
         * @brief Get error message from last failed operation
         *
         * @return Error message, empty string if no error
         */
        virtual const std::string &lastError() const = 0;

        /**
         * @brief Get vocabulary size
         *
         * @return Vocabulary size from loaded model
         */
        virtual int vocabSize() const = 0;

        /**
         * @brief Get current position in KV cache
         *
         * @return Number of tokens processed so far
         */
        virtual int currentPosition() const = 0;

        /**
         * @brief Clear KV cache and reset position
         *
         * Call before starting a new sequence.
         */
        virtual void clearCache() = 0;

        // =====================================================================
        // Advanced
        // =====================================================================

        /**
         * @brief Get logits from last forward pass
         *
         * Returns raw logits before sampling. Shape: [vocab_size]
         *
         * @return Pointer to logits, or nullptr if not available
         */
        virtual const float *lastLogits() const = 0;

        /**
         * @brief Set stop tokens for generation
         *
         * Generation stops when any of these tokens is generated.
         *
         * @param stop_tokens Vector of token IDs that trigger stop
         */
        virtual void setStopTokens(const std::vector<int32_t> &stop_tokens) = 0;

        /**
         * @brief Get the tokenizer
         * @return Shared pointer to tokenizer, or nullptr if not initialized
         */
        virtual std::shared_ptr<ITokenizer> tokenizer() const = 0;

        // =====================================================================
        // Snapshot Capture (for parity testing)
        // =====================================================================

        /**
         * @brief Enable snapshot capture of intermediate activations
         *
         * When enabled, intermediate tensors are captured during forward pass.
         * These can be retrieved via getSnapshot() for comparison with
         * reference implementations (e.g., PyTorch).
         *
         * @param output_dir Optional directory to save snapshots
         */
        virtual void enableSnapshotCapture(const std::string &output_dir = "") = 0;

        /**
         * @brief Disable snapshot capture and clear stored snapshots
         */
        virtual void disableSnapshotCapture() = 0;

        /**
         * @brief Clear stored snapshots but keep capture enabled
         */
        virtual void clearSnapshots() = 0;

        /**
         * @brief Retrieve a captured snapshot by key
         *
         * @param key Snapshot identifier (e.g., "layer0_Q_PROJECTION", "EMBEDDING")
         * @param out_size Output parameter for snapshot size in bytes
         * @return Pointer to snapshot data (FP32), or nullptr if key doesn't exist
         */
        virtual const float *getSnapshot(const std::string &key, size_t &out_size) const = 0;

        /**
         * @brief Get list of all captured snapshot keys
         *
         * @return Vector of snapshot identifiers
         */
        virtual std::vector<std::string> getSnapshotKeys() const = 0;

        // =====================================================================
        // Profiling
        // =====================================================================

        /**
         * @brief Get executor profiling statistics
         *
         * Returns per-stage overhead breakdown (coherence, allocation, etc.)
         * when profiling is enabled (LLAMINAR_PROFILING=1).
         *
         * @return Pointer to GraphExecutorStats, or nullptr if not available
         */
        virtual const GraphExecutorStats *executorStats() const { return nullptr; }

        /**
         * @brief Reset executor profiling statistics
         */
        virtual void resetExecutorStats() {}
    };

} // namespace llaminar2
