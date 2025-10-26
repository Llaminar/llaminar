/**
 * @file PipelineConfig.h
 * @brief Runtime configuration for pipeline initialization
 * @author David Sanftenberg
 * @date 2025-10-25
 *
 * Encapsulates runtime parameters that affect pipeline behavior but are not
 * part of the model architecture (which comes from GGUF metadata).
 */

#pragma once

namespace llaminar2
{

    /**
     * @brief Runtime configuration for pipeline initialization
     *
     * This struct separates runtime configuration (user-specified parameters)
     * from model architecture (GGUF metadata) and device placement (WeightPlacementMap).
     *
     * Design rationale:
     * - ModelContext: Model file metadata (immutable, from GGUF)
     * - WeightPlacementMap: Device placement decisions (Phase 4)
     * - PipelineConfig: Runtime behavior settings (this struct)
     *
     * This separation follows V2's philosophy of clear ownership and composability.
     */
    struct PipelineConfig
    {
        /**
         * @brief Maximum sequence length for inference
         *
         * Determines buffer allocation size for:
         * - KV cache: n_layers × max_seq_len × n_kv_heads × head_dim
         * - Activation buffers: max_seq_len × d_model (various stages)
         *
         * Typical values:
         * - 512: Short conversations, low memory
         * - 2048: Standard conversations (default)
         * - 4096: Long context
         * - 8192+: Extended context models
         *
         * Note: Actual sequence can be shorter, but cannot exceed this limit.
         */
        int max_seq_len = 2048;

        /**
         * @brief Number of OpenMP threads for CPU operations
         *
         * -1 = auto-detect (use all available cores)
         * 0 = single-threaded
         * >0 = explicit thread count
         *
         * Note: Ignored for GPU operations
         */
        int n_threads = -1;

        /**
         * @brief Batch size for batched inference
         *
         * Future extension for batched processing (not yet implemented in V2).
         * Currently must be 1.
         */
        int batch_size = 1;

        /**
         * @brief Enable memory-mapped file access for weights
         *
         * true = mmap weights (lower memory, slower first access)
         * false = load all weights to RAM (higher memory, faster access)
         */
        bool use_mmap = true;

        /**
         * @brief Random seed for sampling
         *
         * -1 = random seed (time-based)
         * ≥0 = deterministic seed for reproducibility
         */
        int seed = -1;

        /**
         * @brief Default constructor with standard settings
         */
        PipelineConfig() = default;

        /**
         * @brief Construct with explicit max_seq_len (common case)
         */
        explicit PipelineConfig(int max_seq_len_) : max_seq_len(max_seq_len_) {}
    };

} // namespace llaminar2
