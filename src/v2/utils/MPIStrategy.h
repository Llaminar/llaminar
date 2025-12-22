/**
 * @file MPIStrategy.h
 * @brief MPI parallelization strategies for distributed inference
 *
 * Defines strategies for splitting transformer operations across MPI ranks.
 * Inspired by V1's DistributionType enum but adapted for V2's operator-free design.
 *
 * @author David Sanftenberg
 */

#pragma once

#include <string>

namespace llaminar2
{

    /**
     * @brief MPI parallelization strategies for distributed inference
     *
     * These strategies determine how transformer operations (attention, FFN)
     * are distributed across multiple MPI ranks for parallel execution.
     */
    enum class MPIStrategy
    {
        /**
         * @brief No parallelization (single rank or disabled)
         *
         * Each rank operates independently with full model weights.
         *
         * Use when:
         * - world_size == 1 (single-rank execution)
         * - User explicitly disabled MPI parallelization
         * - Model architecture incompatible with other strategies
         *
         * Memory: Full model weights on each rank
         * Communication: None
         */
        None,

        /**
         * @brief Tensor parallel - split attention heads/features across ranks
         *
         * Distributes computation by splitting tensor dimensions:
         * - Attention: Each rank computes subset of attention heads
         * - FFN: Each rank computes subset of intermediate features
         *
         * Algorithm (Attention):
         * 1. Rank i computes heads [i*n_heads/world_size, (i+1)*n_heads/world_size)
         * 2. Local Q/K/V projections for assigned heads
         * 3. Local attention computation
         * 4. Allreduce outputs to sum across all ranks
         *
         * Communication: 2× allreduce per layer (after attention, after FFN)
         * Best for: Large models (>7B), high-bandwidth interconnects (InfiniBand, NVLink)
         *
         * Requirements:
         * - n_heads % world_size == 0 (attention heads evenly divisible)
         * - d_ff % world_size == 0 (FFN intermediate dimension evenly divisible)
         *
         * Performance: Near-linear scaling for large models (1.9× on 2 ranks)
         */
        TensorParallel,

        /**
         * @brief Pipeline parallel - split layers across ranks
         *
         * Distributes layers across ranks:
         * - Rank 0: Layers 0 to (n_layers/world_size - 1)
         * - Rank 1: Layers (n_layers/world_size) to (2*n_layers/world_size - 1)
         * - ...
         *
         * Algorithm:
         * 1. Rank i processes its assigned layers sequentially
         * 2. Send activations to rank i+1 via point-to-point communication
         * 3. Receive activations from rank i-1
         *
         * Communication: Point-to-point activation passing between adjacent ranks
         * Best for: Very deep models (>48 layers), lower bandwidth interconnects
         *
         * Requirements:
         * - n_layers % world_size == 0 (layers evenly divisible)
         *
         * Performance: Lower communication overhead, but pipeline bubbles reduce efficiency
         */
        PipelineParallel,

        /**
         * @brief Sequence parallel - split sequence dimension across ranks
         *
         * Distributes tokens across ranks (prefill optimization):
         * - Rank i processes tokens [i*seq_len/world_size, (i+1)*seq_len/world_size)
         *
         * Algorithm:
         * 1. Distribute input tokens across ranks
         * 2. Local embedding lookup
         * 3. Allgather for full sequence before attention
         * 4. Local attention computation
         * 5. Reduce for final output
         *
         * Communication: Allgather for attention inputs, reduce for outputs
         * Best for: Long sequences (>1024 tokens), prefill phase optimization
         *
         * Requirements:
         * - seq_len % world_size == 0 (tokens evenly divisible)
         *
         * Performance: Excellent for prefill, not used for decode (seq_len=1)
         */
        SequenceParallel,

        /**
         * @brief Hybrid - combination of strategies
         *
         * Combines multiple strategies for optimal resource utilization:
         * - Example 1: Tensor-parallel within node + pipeline-parallel across nodes
         * - Example 2: Tensor-parallel for attention + replicated for FFN
         *
         * Requirements: Advanced configuration (not implemented in Phase 1)
         *
         * Performance: Potentially best of all worlds, but complex to configure
         */
        Hybrid
    };

    /**
     * @brief MPI configuration for pipeline execution
     *
     * Controls how MPI parallelization is applied to transformer operations.
     */
    struct MPIConfig
    {
        /**
         * @brief Primary parallelization strategy
         *
         * Default: TensorParallel (most common, best performance for typical models)
         */
        MPIStrategy strategy = MPIStrategy::TensorParallel;

        /**
         * @brief Automatically select strategy based on model architecture
         *
         * If true, ignores 'strategy' field and selects based on:
         * - Model size (n_layers, n_heads, d_model)
         * - MPI world size
         * - Dimension divisibility
         *
         * Selection heuristic:
         * 1. Try TensorParallel if n_heads % world_size == 0
         * 2. Try PipelineParallel if n_layers % world_size == 0
         * 3. Fallback to None (warn user)
         *
         * Default: true (recommended for most users)
         */
        bool auto_select = true;

        /**
         * @brief Validate dimension divisibility before execution
         *
         * If true, checks that chosen strategy is compatible with model:
         * - TensorParallel: n_heads % world_size == 0
         * - PipelineParallel: n_layers % world_size == 0
         *
         * Default: true (fail fast with clear error message)
         */
        bool validate_divisibility = true;

        /**
         * @brief Split attention heads in tensor-parallel mode
         *
         * Default: true (core tensor-parallel optimization)
         */
        bool tp_split_attention = true;

        /**
         * @brief Split FFN intermediate dimension in tensor-parallel mode
         *
         * If true, FFN gate/up projections are also split across ranks.
         * If false, FFN is replicated (each rank computes full FFN).
         *
         * Trade-off:
         * - Split: Lower memory, more communication (allreduce after FFN)
         * - Replicated: Higher memory, less communication
         *
         * Default: true (consistent with attention parallelization)
         */
        bool tp_split_ffn = true;

        /**
         * @brief Fallback strategy if primary fails validation
         *
         * Used when:
         * - auto_select=false and user-specified strategy invalid
         * - Primary strategy fails divisibility check
         *
         * Default: None (disable MPI parallelization rather than incorrect execution)
         */
        MPIStrategy fallback_strategy = MPIStrategy::None;

        /**
         * @brief Enable verbose MPI logging
         *
         * If true, logs:
         * - Strategy selection process
         * - Head/layer/token distribution per rank
         * - Communication timing
         *
         * Default: false (only log warnings/errors)
         */
        bool verbose_logging = false;
    };

    /**
     * @brief Convert MPIStrategy to human-readable string
     *
     * @param strategy Strategy enum value
     * @return String representation (e.g., "TensorParallel")
     */
    inline const char *strategyName(MPIStrategy strategy)
    {
        switch (strategy)
        {
        case MPIStrategy::None:
            return "None";
        case MPIStrategy::TensorParallel:
            return "TensorParallel";
        case MPIStrategy::PipelineParallel:
            return "PipelineParallel";
        case MPIStrategy::SequenceParallel:
            return "SequenceParallel";
        case MPIStrategy::Hybrid:
            return "Hybrid";
        default:
            return "Unknown";
        }
    }

    /**
     * @brief Get default MPI configuration
     *
     * @return Default config with auto_select=true, strategy=TensorParallel
     */
    inline MPIConfig defaultMPIConfig()
    {
        return MPIConfig{}; // All defaults from struct definition
    }

} // namespace llaminar2
