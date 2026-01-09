/**
 * @file PlacementPlan.h
 * @brief Data structures for weight/compute placement decisions
 *
 * PlacementPlan encapsulates the output of a PlacementStrategy:
 * - Which rank owns each layer's weights
 * - Which device (CPU/GPU) executes each layer's compute
 * - Global tensors (embedding, lm_head) placement
 *
 * This separates the "what goes where" decision from the execution.
 * PlacementPlan is computed once at startup (after capability exchange)
 * and then applied to WeightPlacementMap for weight loading.
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#pragma once

#include "../backends/DeviceId.h"
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Device type for placement decisions
     *
     * Simplified enum for placement - actual device index determined
     * by rank-local device enumeration.
     */
    enum class PlacementDevice
    {
        CPU,        ///< Execute on CPU (device_idx = 0)
        GPU_0,      ///< Execute on first GPU (device_idx = 1, if available)
        GPU_1,      ///< Execute on second GPU (device_idx = 2, if available)
        GPU_2,      ///< Execute on third GPU (device_idx = 3, if available)
        GPU_3,      ///< Execute on fourth GPU (device_idx = 4, if available)
        GPU_ANY,    ///< Use any available GPU (resolve at runtime)
        REPLICATED, ///< Replicate on all devices (for small tensors like norms)
    };

    /**
     * @brief Convert PlacementDevice to DeviceId
     * @param device PlacementDevice enum
     * @return DeviceId (CPU or CUDA device)
     */
    inline DeviceId toDeviceId(PlacementDevice device)
    {
        switch (device)
        {
        case PlacementDevice::CPU:
            return DeviceId::cpu();
        case PlacementDevice::GPU_0:
            return DeviceId::cuda(0);
        case PlacementDevice::GPU_1:
            return DeviceId::cuda(1);
        case PlacementDevice::GPU_2:
            return DeviceId::cuda(2);
        case PlacementDevice::GPU_3:
            return DeviceId::cuda(3);
        case PlacementDevice::GPU_ANY:
            return DeviceId::cuda(0); // Default to first GPU
        case PlacementDevice::REPLICATED:
            return DeviceId::cpu(); // Replicated tensors loaded to CPU by default
        }
        return DeviceId::cpu();
    }

    /**
     * @brief Convert PlacementDevice to device index (DEPRECATED)
     * @param device PlacementDevice enum
     * @return Device index (0 = CPU, 1+ = GPUs)
     * @deprecated Use toDeviceId() instead
     */
    [[deprecated("Use toDeviceId() instead")]]
    inline int toDeviceIndex(PlacementDevice device)
    {
        switch (device)
        {
        case PlacementDevice::CPU:
            return 0;
        case PlacementDevice::GPU_0:
            return 1;
        case PlacementDevice::GPU_1:
            return 2;
        case PlacementDevice::GPU_2:
            return 3;
        case PlacementDevice::GPU_3:
            return 4;
        case PlacementDevice::GPU_ANY:
            return 1; // Default to first GPU
        case PlacementDevice::REPLICATED:
            return 0; // Replicated tensors loaded to CPU by default
        }
        return 0;
    }

    /**
     * @brief Placement decision for a single layer's weights/compute
     */
    struct LayerPlacement
    {
        int layer_idx = -1;      ///< Layer index (0-based)
        int owner_rank = 0;      ///< MPI rank that owns this layer's weights (for TP sharding)
        PlacementDevice device = ///< Device for compute
            PlacementDevice::CPU;

        // Fine-grained placement (optional - if attention/FFN on different devices)
        PlacementDevice attention_device = PlacementDevice::CPU;
        PlacementDevice ffn_device = PlacementDevice::CPU;
        bool split_attention_ffn = false; ///< If true, use separate attention_device/ffn_device

        /// Get DeviceId for attention compute
        DeviceId getAttentionDevice() const
        {
            return toDeviceId(split_attention_ffn ? attention_device : device);
        }

        /// Get DeviceId for FFN compute
        DeviceId getFFNDevice() const
        {
            return toDeviceId(split_attention_ffn ? ffn_device : device);
        }

        /// Get device index for attention compute (DEPRECATED)
        [[deprecated("Use getAttentionDevice() instead")]]
        int getAttentionDeviceIdx() const
        {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            return toDeviceIndex(split_attention_ffn ? attention_device : device);
#pragma GCC diagnostic pop
        }

        /// Get device index for FFN compute (DEPRECATED)
        [[deprecated("Use getFFNDevice() instead")]]
        int getFFNDeviceIdx() const
        {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            return toDeviceIndex(split_attention_ffn ? ffn_device : device);
#pragma GCC diagnostic pop
        }
    };

    /**
     * @brief Placement decision for global (non-layer) tensors
     */
    struct GlobalPlacement
    {
        PlacementDevice embedding_device = PlacementDevice::CPU;
        PlacementDevice lm_head_device = PlacementDevice::CPU;
        PlacementDevice final_norm_device = PlacementDevice::CPU;

        /// Whether to shard embedding across ranks (for large vocab)
        bool shard_embedding = false;

        /// Whether to shard lm_head across ranks (for large vocab)
        bool shard_lm_head = false;
    };

    /**
     * @brief Complete placement plan for a model
     *
     * Contains all placement decisions needed to:
     * 1. Load weights to correct devices (via WeightPlacementMap)
     * 2. Route compute to correct devices (via pipeline/orchestrator)
     * 3. Set up MPI communication patterns (which ranks need to communicate)
     *
     * Computed by PlacementStrategy::compute() based on:
     * - Model architecture (n_layers, dimensions)
     * - Available devices across all ranks (from MPITopology)
     * - Optimization goals (memory, latency, throughput)
     */
    struct PlacementPlan
    {
        // =====================================================================
        // Model Info (input to strategy)
        // =====================================================================
        int n_layers = 0;              ///< Total layers in model
        size_t model_memory_bytes = 0; ///< Estimated total model size
        std::string architecture;      ///< Model architecture name (e.g., "qwen2")

        // =====================================================================
        // MPI Topology Info (input to strategy)
        // =====================================================================
        int world_size = 1;          ///< Total MPI ranks
        int ranks_per_node = 1;      ///< Ranks per physical node
        int node_count = 1;          ///< Number of physical nodes
        bool has_gpu = false;        ///< Whether any rank has GPU
        size_t total_gpu_memory = 0; ///< Sum of GPU memory across all ranks

        // =====================================================================
        // Placement Decisions (output of strategy)
        // =====================================================================

        /// Per-layer placement decisions
        std::vector<LayerPlacement> layers;

        /// Global tensor placement
        GlobalPlacement global;

        /// Strategy that generated this plan (for logging/debugging)
        std::string strategy_name;

        // =====================================================================
        // Convenience Queries
        // =====================================================================

        /**
         * @brief Get placement for a specific layer
         * @param layer_idx Layer index (0-based)
         * @return LayerPlacement for that layer
         */
        const LayerPlacement &getLayerPlacement(int layer_idx) const
        {
            static LayerPlacement default_placement;
            if (layer_idx < 0 || layer_idx >= static_cast<int>(layers.size()))
            {
                return default_placement;
            }
            return layers[layer_idx];
        }

        /**
         * @brief Check if any layer uses GPU compute
         */
        bool usesGPU() const
        {
            for (const auto &layer : layers)
            {
                if (layer.device != PlacementDevice::CPU)
                {
                    return true;
                }
            }
            return global.embedding_device != PlacementDevice::CPU ||
                   global.lm_head_device != PlacementDevice::CPU;
        }

        /**
         * @brief Check if tensor parallelism is used (weights sharded across ranks)
         */
        bool usesTensorParallelism() const
        {
            return world_size > 1;
        }

        /**
         * @brief Get list of ranks that own at least one layer
         */
        std::vector<int> getActiveRanks() const
        {
            std::vector<bool> active(world_size, false);
            for (const auto &layer : layers)
            {
                if (layer.owner_rank >= 0 && layer.owner_rank < world_size)
                {
                    active[layer.owner_rank] = true;
                }
            }
            std::vector<int> result;
            for (int r = 0; r < world_size; ++r)
            {
                if (active[r])
                {
                    result.push_back(r);
                }
            }
            return result;
        }

        /**
         * @brief Get device index for attention in a layer (for this rank)
         * @param layer_idx Layer index
         * @return Device index (0 = CPU, 1+ = GPU)
         */
        int getAttentionDevice(int layer_idx) const
        {
            return getLayerPlacement(layer_idx).getAttentionDeviceIdx();
        }

        /**
         * @brief Get device index for FFN in a layer (for this rank)
         * @param layer_idx Layer index
         * @return Device index (0 = CPU, 1+ = GPU)
         */
        int getFFNDevice(int layer_idx) const
        {
            return getLayerPlacement(layer_idx).getFFNDeviceIdx();
        }

        /**
         * @brief Check if this plan is valid (all fields populated correctly)
         */
        bool isValid() const
        {
            if (n_layers <= 0 || world_size <= 0)
            {
                return false;
            }
            if (layers.size() != static_cast<size_t>(n_layers))
            {
                return false;
            }
            for (const auto &layer : layers)
            {
                if (layer.owner_rank < 0 || layer.owner_rank >= world_size)
                {
                    return false;
                }
            }
            return true;
        }

        /**
         * @brief Generate human-readable summary of the plan
         */
        std::string toString() const;
    };

} // namespace llaminar2
