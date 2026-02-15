/**
 * @file PlacementStrategy.h
 * @brief Strategy interface for computing weight/compute placement
 *
 * PlacementStrategy computes a PlacementPlan from:
 * - Model info (architecture, n_layers, memory estimates)
 * - MPI topology (ranks, nodes, device capabilities)
 * - ClusterInventory (all devices across all ranks)
 *
 * Available strategies:
 * - CPUOnlyPlacementStrategy: All compute on CPU (baseline)
 * - GPUFirstPlacementStrategy: Fill GPU memory first, spill to CPU when exhausted
 * - HybridOptimalPlacementStrategy: Optimize placement based on prefill vs decode characteristics
 *
 * Strategy selection:
 * - Automatic: Based on available devices and model size
 * - Manual: User specifies via CLI flag or env var
 *
 * Design principle: All ranks compute the SAME placement plan deterministically.
 * This avoids needing to broadcast the plan - each rank runs the same algorithm
 * with the same inputs (post-capability-exchange) and gets the same output.
 *
 * Hierarchical placement workflow:
 * 1. Each rank discovers its local devices (GPUs, CPU cores, memory)
 * 2. MPI_Allgather exchanges inventories to build ClusterInventory
 * 3. All ranks run the same PlacementStrategy::compute() with identical input
 * 4. Each rank extracts its portion of the placement plan
 * 5. Tensor parallelism happens within rank (across local devices) and across ranks
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#pragma once

#include "PlacementPlan.h"
#include "DeviceInventory.h"
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class MPITopology;
    struct RankPlacement;
    /**
     * @brief Inference phase for phase-aware placement decisions
     *
     * Different phases have fundamentally different performance characteristics:
     * - PREFILL: Compute-bound (large batch matmuls) → GPUs strongly preferred
     * - DECODE: Memory-bandwidth-bound (single-token) → ALL devices including CPU
     *
     * Key insight: During decode, every memory channel matters. Modern Xeons with
     * 6-8 memory channels provide 230-360 GB/s bandwidth - significant and should
     * not be wasted as mere staging buffers.
     */
    enum class InferencePhase
    {
        PREFILL, ///< Processing prompt tokens (compute-bound, GPUs only)
        DECODE   ///< Generating tokens (bandwidth-bound, all devices)
    };

    /**
     * @brief Convert InferencePhase to string for logging
     */
    inline const char *toString(InferencePhase phase)
    {
        switch (phase)
        {
        case InferencePhase::PREFILL:
            return "PREFILL";
        case InferencePhase::DECODE:
            return "DECODE";
        default:
            return "UNKNOWN";
        }
    }

    /**
     * @brief Input parameters for placement strategy computation
     *
     * Contains all information needed to compute a PlacementPlan.
     * Gathered from ModelLoader metadata and MPITopology.
     *
     * The cluster_inventory is the key field for hierarchical placement.
     * It contains complete device information for all ranks.
     */
    struct PlacementInput
    {
        // =====================================================================
        // Model info
        // =====================================================================
        std::string architecture;                ///< e.g., "qwen2", "llama"
        int n_layers = 0;                        ///< Number of transformer layers
        size_t d_model = 0;                      ///< Hidden dimension
        size_t d_ff = 0;                         ///< FFN intermediate dimension
        size_t vocab_size = 0;                   ///< Vocabulary size
        size_t n_heads = 0;                      ///< Attention heads
        size_t n_kv_heads = 0;                   ///< KV heads (for GQA)
        std::string kv_cache_precision = "auto"; ///< KV cache precision for memory estimates: "auto", "fp16", "q8_1"
        std::string quant_type;                  ///< Quantization type (e.g., "Q4_0", "Q8_0")
        size_t estimated_memory_bytes = 0;       ///< Estimated total model memory

        // =====================================================================
        // Cluster topology (from ClusterInventory after capability exchange)
        // =====================================================================
        int world_size = 1;                      ///< Total MPI ranks
        int ranks_per_node = 1;                  ///< Ranks per physical node
        int node_count = 1;                      ///< Number of physical nodes
        std::vector<float> rank_compute_weights; ///< Relative compute power per rank

        // Complete cluster device inventory (populated by MPITopology::exchangeCapabilities)
        ClusterInventory cluster_inventory;

        // =====================================================================
        // Aggregated device info (convenience, derived from cluster_inventory)
        // =====================================================================
        bool any_rank_has_gpu = false; ///< Any rank has GPU?
        size_t total_gpu_memory = 0;   ///< Total GPU memory across cluster
        size_t total_cpu_memory = 0;   ///< Total CPU memory across cluster
        int total_gpu_count = 0;       ///< Total GPUs across cluster

        // Per-rank GPU info (for backward compatibility)
        // DEPRECATED: Use cluster_inventory.ranks[rank].gpus instead
        std::vector<size_t> gpu_memory_per_device; ///< GPU memory per device (rank-local view)

        // =====================================================================
        // Performance characteristics (for HybridOptimalPlacementStrategy)
        // =====================================================================
        float cpu_memory_bandwidth_gbps = 0.0f; ///< CPU DRAM bandwidth (e.g., 200 GB/s for Xeon)
        float gpu_memory_bandwidth_gbps = 0.0f; ///< GPU HBM bandwidth (e.g., 2000 GB/s for H100)
        float cpu_compute_tflops = 0.0f;        ///< CPU compute (e.g., 2 TFLOPS for AVX-512)
        float gpu_compute_tflops = 0.0f;        ///< GPU compute (e.g., 1000 TFLOPS for H100)

        // =====================================================================
        // Constraints (from CLI or environment)
        // =====================================================================
        bool force_cpu_only = false;    ///< --cpu-only flag
        bool force_gpu_only = false;    ///< --gpu-only flag
        int max_gpu_layers = -1;        ///< --n-gpu-layers N (-1 = no limit)
        std::string preferred_strategy; ///< User-specified strategy name

        // =====================================================================
        // Convenience fields for simple GPU availability testing
        // (These are derived from cluster_inventory but provided for simpler APIs)
        // =====================================================================
        bool has_gpu = false;        ///< Local rank has at least one GPU available
        size_t gpu_memory_bytes = 0; ///< Total GPU memory on local rank (bytes)

        // =====================================================================
        // Helper methods
        // =====================================================================

        /// Get GPU count for a specific rank
        int gpuCountForRank(int rank) const
        {
            if (rank >= 0 && rank < static_cast<int>(cluster_inventory.ranks.size()))
            {
                return cluster_inventory.ranks[rank].gpuCount();
            }
            return 0;
        }

        /// Get total GPU memory for a specific rank
        size_t gpuMemoryForRank(int rank) const
        {
            if (rank >= 0 && rank < static_cast<int>(cluster_inventory.ranks.size()))
            {
                return cluster_inventory.ranks[rank].totalGPUMemory();
            }
            return 0;
        }

        /// Get GPU memory for a specific device on a rank
        size_t gpuMemoryForDevice(int rank, int device_idx) const
        {
            if (rank >= 0 && rank < static_cast<int>(cluster_inventory.ranks.size()))
            {
                const auto &gpus = cluster_inventory.ranks[rank].gpus;
                if (device_idx >= 0 && device_idx < static_cast<int>(gpus.size()))
                {
                    return gpus[device_idx].memory_bytes;
                }
            }
            return 0;
        }

        /// Populate aggregated fields from cluster_inventory
        void updateAggregatedFields()
        {
            any_rank_has_gpu = cluster_inventory.hasAnyGPU();
            total_gpu_memory = cluster_inventory.total_gpu_memory;
            total_cpu_memory = cluster_inventory.total_cpu_memory;
            total_gpu_count = cluster_inventory.total_gpus;
            world_size = cluster_inventory.world_size;
            node_count = cluster_inventory.node_count;
        }

        // =====================================================================
        // Phase-Aware Device Selection (CPU as first-class decode participant)
        // =====================================================================

        /**
         * @brief Get device weights for a specific inference phase
         *
         * Key insight: Prefill is compute-bound (GPU dominates), but decode is
         * bandwidth-bound (all memory channels matter, including CPU).
         *
         * For PREFILL: GPUs get all the work (high FLOPS utilization)
         * For DECODE: Work distributed by memory bandwidth (CPU included!)
         *
         * @param phase Inference phase (PREFILL or DECODE)
         * @return Weights per device type [GPU_weight, CPU_weight]
         */
        std::pair<float, float> getPhaseDeviceWeights(InferencePhase phase) const
        {
            if (phase == InferencePhase::PREFILL || !any_rank_has_gpu)
            {
                // Prefill: GPU gets 100% (compute-bound)
                // Or no GPU: CPU gets 100%
                return any_rank_has_gpu ? std::make_pair(1.0f, 0.0f) : std::make_pair(0.0f, 1.0f);
            }

            // DECODE: Distribute by memory bandwidth
            float total_bw = gpu_memory_bandwidth_gbps + cpu_memory_bandwidth_gbps;
            if (total_bw <= 0.0f)
            {
                // No bandwidth info: use heuristic (GPU ~5x CPU for decode)
                return std::make_pair(0.83f, 0.17f);
            }

            float gpu_weight = gpu_memory_bandwidth_gbps / total_bw;
            float cpu_weight = cpu_memory_bandwidth_gbps / total_bw;
            return std::make_pair(gpu_weight, cpu_weight);
        }

        /**
         * @brief Check if CPU should participate in a given phase
         *
         * @param phase Inference phase
         * @return true if CPU should do compute work (not just staging)
         */
        bool cpuShouldParticipate(InferencePhase phase) const
        {
            if (!any_rank_has_gpu)
                return true; // No GPUs: CPU must participate

            if (phase == InferencePhase::PREFILL)
                return false; // Prefill: GPU-only (compute-bound)

            // DECODE: CPU participates if it has meaningful bandwidth
            // Threshold: CPU should have at least 10% of total bandwidth
            if (cpu_memory_bandwidth_gbps > 0.0f && gpu_memory_bandwidth_gbps > 0.0f)
            {
                float cpu_fraction = cpu_memory_bandwidth_gbps /
                                     (cpu_memory_bandwidth_gbps + gpu_memory_bandwidth_gbps);
                return cpu_fraction >= 0.05f; // 5% threshold (very inclusive)
            }

            // No bandwidth info: conservative - include CPU for decode
            return true;
        }

        /**
         * @brief Get total memory bandwidth available for decode
         *
         * Includes ALL devices (GPUs + CPUs) since decode is bandwidth-bound.
         *
         * @return Total bandwidth in GB/s
         */
        float getTotalDecodeBandwidth() const
        {
            return gpu_memory_bandwidth_gbps + cpu_memory_bandwidth_gbps;
        }
    };

    /**
     * @brief Abstract base class for layer placement strategies
     *
     * Subclasses implement compute() to generate a PlacementPlan
     * based on model info and device capabilities.
     *
     * NOTE: This class controls where COMPUTE runs (layers on CPU vs GPU).
     * For weight storage placement, see WeightPlacementStrategy enum in DeviceOrchestrator.h.
     */
    class LayerPlacementStrategy
    {
    public:
        virtual ~LayerPlacementStrategy() = default;

        /**
         * @brief Compute a placement plan from inputs
         * @param input Model and topology information
         * @return Computed placement plan
         *
         * IMPORTANT: This method must be DETERMINISTIC. All ranks call it
         * with the same inputs (after capability exchange) and must get
         * the exact same output. This avoids needing to broadcast the plan.
         */
        virtual PlacementPlan compute(const PlacementInput &input) const = 0;

        /**
         * @brief Get strategy name for logging/debugging
         */
        virtual std::string name() const = 0;

        /**
         * @brief Check if this strategy is applicable to the given input
         * @param input Model and topology information
         * @return true if strategy can generate a valid plan
         *
         * Used by automatic strategy selection to find applicable strategies.
         */
        virtual bool isApplicable(const PlacementInput &input) const = 0;
    };

    /**
     * @brief CPU-only layer placement strategy: All compute on CPU
     *
     * This is the baseline strategy. All layers execute on CPU.
     * Weights are distributed across ranks for tensor parallelism,
     * but no GPU compute is used.
     *
     * Use cases:
     * - Systems without GPU
     * - Debugging/testing without GPU complexity
     * - When model fits entirely in CPU memory with good DRAM bandwidth
     *
     * Applicable when:
     * - force_cpu_only is true, OR
     * - No GPU is available on any rank, OR
     * - User explicitly selects this strategy
     */
    class CPUOnlyLayerPlacementStrategy : public LayerPlacementStrategy
    {
    public:
        PlacementPlan compute(const PlacementInput &input) const override;
        std::string name() const override { return "CPUOnly"; }
        bool isApplicable(const PlacementInput &input) const override;
    };

    /**
     * @brief GPU-first layer placement strategy: Maximize GPU utilization
     *
     * Places as many layers on GPU as memory allows, then spills
     * remaining layers to CPU. Simple greedy approach that prioritizes
     * GPU utilization. Supports distributing across multiple GPUs.
     *
     * Algorithm:
     * 1. Estimate memory per layer (packed weights + KV cache + activations)
     * 2. Distribute layers across available GPUs until memory exhausted
     * 3. Place remaining layers on CPU
     * 4. Global tensors (embedding, lm_head) on GPU if they fit
     *
     * Memory estimation accounts for:
     * - CUDA INT8 repacking (K×N bytes + N×4 scale bytes)
     * - Activation buffers (QKV projections, attention output, FFN buffers)
     * - KV cache per layer
     *
     * Use cases:
     * - GPU with limited memory (consumer GPUs)
     * - Multi-GPU systems (distributes across GPUs)
     * - When GPU compute advantage outweighs CPU DRAM bandwidth
     *
     * Applicable when:
     * - At least one GPU is available
     * - force_cpu_only is false
     */
    class GPUFirstLayerPlacementStrategy : public LayerPlacementStrategy
    {
    public:
        PlacementPlan compute(const PlacementInput &input) const override;
        std::string name() const override { return "GPUFirst"; }
        bool isApplicable(const PlacementInput &input) const override;

    private:
        /**
         * @brief Estimate memory required for one transformer layer on a device
         * @param input Model parameters
         * @param device Target device (affects packed weight format)
         * @return Memory in bytes for weights + KV cache + activation buffers
         */
        size_t estimateLayerMemory(const PlacementInput &input, PlacementDevice device) const;

        /**
         * @brief Estimate memory for global tensors (embedding, lm_head)
         * @param input Model parameters
         * @param device Target device
         * @return Memory in bytes
         */
        size_t estimateGlobalMemory(const PlacementInput &input, PlacementDevice device) const;
    };

    /**
     * @brief Hybrid optimal placement strategy: Optimize for prefill AND decode
     *
     * This strategy recognizes that prefill and decode have different characteristics:
     *
     * PREFILL (processing prompt, seq_len >> 1):
     * - Compute-bound: Large matrix multiplications dominate
     * - GPU strongly preferred: High FLOPS utilization
     * - Memory bandwidth less critical: Batch amortizes memory access
     *
     * DECODE (generating tokens, seq_len = 1):
     * - Memory-bound: Weights read once per token, little compute reuse
     * - CPU can be competitive: High DRAM bandwidth (e.g., 200 GB/s on Xeon)
     * - GPU HBM advantage reduced: Single-token batches underutilize compute
     *
     * Algorithm:
     * 1. Calculate memory-bound throughput for CPU (DRAM bandwidth / packed bytes per layer)
     * 2. Calculate memory-bound throughput for GPU (HBM bandwidth / packed bytes per layer)
     * 3. For decode: If CPU memory bandwidth competitive, consider CPU placement
     * 4. For prefill: Strongly prefer GPU for compute-heavy phases
     * 5. Split layers optimally across available GPUs and CPU
     *
     * Memory estimation accounts for device-specific packing:
     * - CPU: VNNI INT8 format (~1.25-1.375 bytes per weight)
     * - GPU: Column-major INT8 (~1.0 bytes per weight + per-column scales)
     *
     * Example (Xeon 8490H + RTX 4090):
     * - CPU: 307 GB/s DDR5 bandwidth, ~2 TFLOPS AVX-512
     * - GPU: 1000 GB/s HBM, 330 TFLOPS FP16
     * - For Q4_0 model decode: CPU decode can achieve ~40% of GPU due to bandwidth
     *
     * Use cases:
     * - High-end Xeon systems with substantial DRAM bandwidth
     * - Multi-GPU systems where memory is limited
     * - Maximizing throughput by utilizing both CPU and GPU
     *
     * Applicable when:
     * - At least one GPU is available
     * - CPU memory bandwidth info is provided (optional but helps)
     * - force_cpu_only and force_gpu_only are both false
     */
    class HybridOptimalLayerPlacementStrategy : public LayerPlacementStrategy
    {
    public:
        PlacementPlan compute(const PlacementInput &input) const override;
        std::string name() const override { return "HybridOptimal"; }
        bool isApplicable(const PlacementInput &input) const override;

    private:
        /**
         * @brief Calculate effective throughput for a layer on CPU (decode)
         * @param input Model parameters
         * @return Tokens per second estimate
         */
        float estimateCPUDecodeTokensPerSec(const PlacementInput &input) const;

        /**
         * @brief Calculate effective throughput for a layer on GPU (decode)
         * @param input Model parameters
         * @return Tokens per second estimate
         */
        float estimateGPUDecodeTokensPerSec(const PlacementInput &input) const;

        /**
         * @brief Determine optimal number of layers on GPU
         * @param input Model parameters
         * @param gpu_memories Available memory per GPU
         * @return Number of layers to place on GPU (rest go to CPU)
         */
        int computeOptimalGPULayers(const PlacementInput &input,
                                    const std::vector<size_t> &gpu_memories) const;

        /**
         * @brief Estimate bytes read per token for a layer (decode phase)
         * @param input Model parameters
         * @param device Target device (affects packed format)
         * @return Bytes per token
         */
        size_t estimateBytesPerToken(const PlacementInput &input, PlacementDevice device) const;
    };

    // =========================================================================
    // Legacy Aliases (for backward compatibility)
    // =========================================================================

    /// @deprecated Use CPUOnlyLayerPlacementStrategy instead
    using CPUOnlyPlacementStrategy = CPUOnlyLayerPlacementStrategy;

    /// @deprecated Use GPUFirstLayerPlacementStrategy instead
    using GPUFirstPlacementStrategy = GPUFirstLayerPlacementStrategy;

    /// @deprecated Use HybridOptimalLayerPlacementStrategy instead
    using HybridOptimalPlacementStrategy = HybridOptimalLayerPlacementStrategy;

    /// @deprecated Use LayerPlacementStrategy instead
    using PlacementStrategy = LayerPlacementStrategy;

    /// @deprecated Legacy alias
    using CPUOnlyStrategy = CPUOnlyLayerPlacementStrategy;

    /// @deprecated Legacy alias
    using GPUFirstStrategy = GPUFirstLayerPlacementStrategy;

    /**
     * @brief Factory for creating layer placement strategies
     *
     * Supports:
     * - Automatic selection based on capabilities
     * - Manual selection by name
     */
    class LayerPlacementStrategyFactory
    {
    public:
        /**
         * @brief Create strategy by name
         * @param name Strategy name ("CPUOnly", "GPUFirst", "HybridOptimal")
         * @return Strategy instance, or nullptr if unknown name
         */
        static std::unique_ptr<LayerPlacementStrategy> create(const std::string &name);

        /**
         * @brief Auto-select best strategy for given input
         * @param input Model and topology information
         * @return Best applicable strategy
         *
         * Selection priority:
         * 1. User-specified strategy (input.preferred_strategy)
         * 2. Force flags (force_cpu_only, force_gpu_only)
         * 3. HybridOptimal if GPU available and bandwidth info provided
         * 4. GPUFirst if GPU available
         * 5. CPUOnly as fallback
         */
        static std::unique_ptr<LayerPlacementStrategy> autoSelect(const PlacementInput &input);

        /**
         * @brief Get list of all available strategy names
         */
        static std::vector<std::string> availableStrategies();
    };

    /// @deprecated Use LayerPlacementStrategyFactory instead
    using PlacementStrategyFactory = LayerPlacementStrategyFactory;

} // namespace llaminar2
