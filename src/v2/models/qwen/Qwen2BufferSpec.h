/**
 * @file Qwen2BufferSpec.h
 * @brief Buffer allocation specifications for Qwen2 models
 * @author David Sanftenberg
 * @date January 2026
 *
 * Qwen2BufferSpec provides utilities for determining buffer allocation
 * strategies based on tensor parallelism configuration and collective
 * backend type. This is particularly important for heterogeneous GPU
 * setups using PCIe BAR for cross-vendor communication.
 *
 * Key Use Cases:
 * - Determining which buffers need BAR-backed allocation for PCIeBAR allreduce
 * - Supporting LOCAL TP with mixed CUDA/ROCm GPUs
 *
 * Row-Parallel Outputs:
 * - FFN down projection: row-parallel, needs allreduce
 * - Attention Wo projection: row-parallel, needs allreduce
 *
 * These buffers benefit from BAR-backed allocation when using PCIeBAR
 * backend, as it enables direct P2P access without staging through host.
 */

#pragma once

#include "../../config/OrchestrationConfig.h"
#include <string>
#include <unordered_set>

namespace llaminar2
{

    // =========================================================================
    // AllocationStrategy Enum
    // =========================================================================

    /**
     * @brief Memory allocation strategy for activation buffers
     *
     * Determines how buffers are allocated based on their usage patterns
     * and the collective communication backend being used.
     */
    enum class AllocationStrategy
    {
        STANDARD,   ///< Regular device memory (VRAM on GPU, system RAM on CPU)
        BAR_BACKED, ///< PCIe BAR-backed for cross-vendor allreduce (ROCm ↔ CUDA)
        PINNED_HOST ///< Pinned host memory for CPU staging (future)
    };

    // =========================================================================
    // Qwen2BufferSpec Class
    // =========================================================================

    /**
     * @brief Buffer specification utilities for Qwen2 models
     *
     * Provides static methods for determining allocation strategies
     * for various buffers based on parallelism configuration.
     */
    class Qwen2BufferSpec
    {
    public:
        /**
         * @brief Check if a buffer needs BAR-backed allocation
         *
         * Returns true for row-parallel output buffers when:
         * - LOCAL TP is enabled (tp_degree > 1)
         * - Backend is PCIeBAR (heterogeneous GPU setup)
         *
         * Affected buffers (exact match):
         * - "ffn_down_output" / "ffn_down_allreduce" - FFN down projection result
         * - "ffn_output" - FFN output (same buffer, different name contexts)
         * - "attention_wo_output" / "attn_wo_output" / "attn_wo_allreduce" - Attention Wo output
         * - "wo_output" / "attn_proj" - Attention projection output
         *
         * Also matches layer-prefixed names via suffix matching:
         * - "layer0_ffn_down_output", "layer5_attn_wo_allreduce", etc.
         *
         * @param buffer_name Name of the buffer (exact or with layer prefix)
         * @param backend_type Collective backend type (PCIE_BAR triggers BAR allocation)
         * @param tp_degree Tensor parallelism degree (1 = no TP, no BAR needed)
         * @return true if buffer should use BAR-backed allocation
         */
        static bool requiresBARBacked(
            const std::string &buffer_name,
            CollectiveBackendType backend_type,
            int tp_degree);

        /**
         * @brief Get the allocation strategy for a buffer
         *
         * Convenience method that returns the appropriate AllocationStrategy
         * enum value based on buffer name and configuration.
         *
         * @param buffer_name Name of the buffer
         * @param backend_type Collective backend type
         * @param tp_degree Tensor parallelism degree
         * @return AllocationStrategy for this buffer
         */
        static AllocationStrategy getAllocationStrategy(
            const std::string &buffer_name,
            CollectiveBackendType backend_type,
            int tp_degree);

        /**
         * @brief Get the set of row-parallel output buffer suffixes
         *
         * These are the buffer name patterns that identify row-parallel
         * outputs needing allreduce. Used for both exact and suffix matching.
         *
         * @return Set of buffer name suffixes for row-parallel outputs
         */
        static const std::unordered_set<std::string> &getRowParallelOutputSuffixes();

    private:
        /**
         * @brief Check if buffer name ends with any of the row-parallel suffixes
         *
         * @param buffer_name Full buffer name (may include layer prefix)
         * @return true if name matches a row-parallel output pattern
         */
        static bool matchesRowParallelOutput(const std::string &buffer_name);
    };

} // namespace llaminar2
