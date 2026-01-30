/**
 * @file ExecutionPolicy.h
 * @brief Declarative control over which operations are enabled during inference
 * @author David Sanftenberg
 * @date December 2025
 *
 * ExecutionPolicy replaces scattered environment variable checks with an explicit,
 * type-safe configuration object. This improves testability, documentation, and
 * makes operation control first-class citizens rather than hidden dependencies.
 *
 * ## Design Rationale
 *
 * Before: Hidden environment variable dependencies scattered throughout code:
 * ```cpp
 * // Hard to discover, no IDE completion, runtime-only validation
 * if (debugEnv().execution.exec_rmsnorm) { ... }
 * ```
 *
 * After: Explicit policy object passed through the call chain:
 * ```cpp
 * ExecutionPolicy policy = ExecutionPolicy::allEnabled();
 * policy.rmsnorm = false;  // Disable RMSNorm for testing
 * executor.execute(layer, buffers, policy);
 * ```
 *
 * ## Usage Patterns
 *
 * 1. **Production (default)**: All operations enabled
 *    ```cpp
 *    auto policy = ExecutionPolicy::allEnabled();
 *    ```
 *
 * 2. **Testing**: Selective operation control
 *    ```cpp
 *    auto policy = ExecutionPolicy::allEnabled();
 *    policy.attention = false;  // Skip attention for FFN-only test
 *    ```
 *
 * 3. **Incremental Migration**: Enable from environment (backward compat)
 *    ```cpp
 *    auto policy = ExecutionPolicy::fromEnvironment();
 *    ```
 *
 * 4. **Debugging**: No-op mode for baseline comparison
 *    ```cpp
 *    auto policy = ExecutionPolicy::noop();
 *    ```
 */

#pragma once

#include <string>
#include <sstream>
#include "../../utils/DebugEnv.h"

namespace llaminar2
{

    /**
     * @brief Declarative execution policy for layer/pipeline operations
     *
     * Each flag controls whether a specific operation type is executed.
     * When a flag is false, the operation is skipped (no-op).
     *
     * Thread Safety: ExecutionPolicy is immutable after construction.
     * Copy/move semantics allow safe passing between threads.
     */
    struct ExecutionPolicy
    {
        // ===== Operation Flags =====
        // Each flag controls whether that operation type executes or is skipped

        bool rmsnorm = true;   ///< Enable RMSNorm operations
        bool rope = true;      ///< Enable RoPE (rotary position embedding)
        bool attention = true; ///< Enable attention computation
        bool gemm = true;      ///< Enable GEMM (matrix multiply) operations
        bool swiglu = true;    ///< Enable SwiGLU activation
        bool residual = true;  ///< Enable residual connections

        // ===== Factory Methods =====

        /**
         * @brief Create policy with all operations enabled (production default)
         * @return ExecutionPolicy with all flags = true
         */
        static ExecutionPolicy allEnabled()
        {
            return ExecutionPolicy{
                .rmsnorm = true,
                .rope = true,
                .attention = true,
                .gemm = true,
                .swiglu = true,
                .residual = true};
        }

        /**
         * @brief Create policy with all operations disabled (for testing/debugging)
         * @return ExecutionPolicy with all flags = false
         */
        static ExecutionPolicy noop()
        {
            return ExecutionPolicy{
                .rmsnorm = false,
                .rope = false,
                .attention = false,
                .gemm = false,
                .swiglu = false,
                .residual = false};
        }

        /**
         * @brief Create policy from environment variables (backward compatibility)
         *
         * Reads LLAMINAR_EXEC_* environment variables via debugEnv().execution:
         *   - LLAMINAR_EXEC_RMSNORM
         *   - LLAMINAR_EXEC_ROPE
         *   - LLAMINAR_EXEC_ATTENTION
         *   - LLAMINAR_EXEC_GEMM
         *   - LLAMINAR_EXEC_SWIGLU
         *   - LLAMINAR_EXEC_RESIDUAL
         *
         * Note: Each flag is independent. Defaults are true (all operations enabled).
         * Set individual flags to 0 to disable specific operations for debugging.
         *
         * @return ExecutionPolicy with flags set from environment
         */
        static ExecutionPolicy fromEnvironment()
        {
            const auto &exec = debugEnv().execution;

            return ExecutionPolicy{
                .rmsnorm = exec.exec_rmsnorm,
                .rope = exec.exec_rope,
                .attention = exec.exec_attention,
                .gemm = exec.exec_gemm,
                .swiglu = exec.exec_swiglu,
                .residual = exec.exec_residual};
        }

        /**
         * @brief Create FFN-only policy (attention disabled)
         *
         * Useful for testing FFN operations in isolation:
         * - RMSNorm: enabled (pre-FFN norm)
         * - SwiGLU: enabled
         * - GEMM: enabled (gate/up/down projections)
         * - Residual: enabled (FFN output addition)
         * - RoPE: disabled (attention only)
         * - Attention: disabled
         *
         * @return ExecutionPolicy configured for FFN-only execution
         */
        static ExecutionPolicy ffnOnly()
        {
            return ExecutionPolicy{
                .rmsnorm = true,
                .rope = false,
                .attention = false,
                .gemm = true,
                .swiglu = true,
                .residual = true};
        }

        /**
         * @brief Create attention-only policy (FFN disabled)
         *
         * Useful for testing attention operations in isolation:
         * - RMSNorm: enabled (pre-attention norm)
         * - RoPE: enabled
         * - Attention: enabled
         * - GEMM: enabled (Q/K/V/O projections)
         * - Residual: enabled (attention output addition)
         * - SwiGLU: disabled (FFN only)
         *
         * @return ExecutionPolicy configured for attention-only execution
         */
        static ExecutionPolicy attentionOnly()
        {
            return ExecutionPolicy{
                .rmsnorm = true,
                .rope = true,
                .attention = true,
                .gemm = true,
                .swiglu = false,
                .residual = true};
        }

        // ===== Utility Methods =====

        /**
         * @brief Check if all operations are enabled
         * @return true if all flags are true
         */
        bool isFullyEnabled() const
        {
            return rmsnorm && rope && attention && gemm && swiglu && residual;
        }

        /**
         * @brief Check if all operations are disabled
         * @return true if all flags are false
         */
        bool isNoop() const
        {
            return !rmsnorm && !rope && !attention && !gemm && !swiglu && !residual;
        }

        /**
         * @brief Get string representation for logging
         * @return Human-readable policy description
         */
        std::string toString() const
        {
            std::ostringstream oss;
            oss << "ExecutionPolicy{";
            oss << "rmsnorm=" << (rmsnorm ? "1" : "0");
            oss << ", rope=" << (rope ? "1" : "0");
            oss << ", attention=" << (attention ? "1" : "0");
            oss << ", gemm=" << (gemm ? "1" : "0");
            oss << ", swiglu=" << (swiglu ? "1" : "0");
            oss << ", residual=" << (residual ? "1" : "0");
            oss << "}";
            return oss.str();
        }

        // ===== Comparison Operators =====

        bool operator==(const ExecutionPolicy &other) const
        {
            return rmsnorm == other.rmsnorm &&
                   rope == other.rope &&
                   attention == other.attention &&
                   gemm == other.gemm &&
                   swiglu == other.swiglu &&
                   residual == other.residual;
        }

        bool operator!=(const ExecutionPolicy &other) const
        {
            return !(*this == other);
        }
    };

} // namespace llaminar2
