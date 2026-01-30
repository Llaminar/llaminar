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
 * Key concept: Placement is hierarchical:
 * - ClusterPlacement: Global view (rank 0 computes, all ranks have identical copy)
 * - RankPlacement: What this rank should do (extracted from ClusterPlacement)
 * - LayerPlacement: Where each layer's compute happens
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#pragma once

#include "../../backends/DeviceId.h"
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Device specification for placement decisions
     *
     * Supports arbitrary GPU counts (up to 32 per rank).
     * CPU is always device index 0, GPUs are 1+.
     */
    struct PlacementDevice
    {
        enum Type
        {
            CPU = 0,        ///< Execute on CPU
            GPU = 1,        ///< Execute on GPU (use gpu_index for which one)
            ANY_GPU = 2,    ///< Use any available GPU (resolve at runtime)
            REPLICATED = 3, ///< Replicate on all devices
        };

        Type type = CPU;
        int gpu_index = 0; ///< GPU index within rank (0-31), only valid if type == GPU

        /// Construct CPU placement
        static PlacementDevice cpu() { return {CPU, 0}; }

        /// Construct GPU placement with specific index
        static PlacementDevice gpu(int index) { return {GPU, index}; }

        /// Construct any-GPU placement
        static PlacementDevice anyGpu() { return {ANY_GPU, 0}; }

        /// Construct replicated placement
        static PlacementDevice replicated() { return {REPLICATED, 0}; }

        /// Check if this is a CPU placement
        bool isCPU() const { return type == CPU; }

        /// Check if this is a GPU placement
        bool isGPU() const { return type == GPU || type == ANY_GPU; }

        /// Get effective GPU index (0 if CPU or ANY_GPU)
        int effectiveGpuIndex() const { return type == GPU ? gpu_index : 0; }

        /// Comparison operators
        bool operator==(const PlacementDevice &other) const
        {
            return type == other.type && gpu_index == other.gpu_index;
        }
        bool operator!=(const PlacementDevice &other) const { return !(*this == other); }

        /// Convert to string for logging
        std::string toString() const
        {
            switch (type)
            {
            case CPU:
                return "CPU";
            case GPU:
                return "GPU_" + std::to_string(gpu_index);
            case ANY_GPU:
                return "GPU_ANY";
            case REPLICATED:
                return "REPLICATED";
            }
            return "UNKNOWN";
        }
    };

    // Legacy compatibility aliases
    namespace LegacyPlacement
    {
        constexpr auto CPU = PlacementDevice::CPU;
        constexpr auto GPU_0 = PlacementDevice::GPU;
        constexpr auto GPU_1 = PlacementDevice::GPU;
        constexpr auto GPU_2 = PlacementDevice::GPU;
        constexpr auto GPU_3 = PlacementDevice::GPU;
        constexpr auto GPU_ANY = PlacementDevice::ANY_GPU;
        constexpr auto REPLICATED = PlacementDevice::REPLICATED;
    }

    /**
     * @brief Convert PlacementDevice to DeviceId
     * @param device PlacementDevice struct
     * @return DeviceId (CPU or CUDA device)
     */
    inline DeviceId toDeviceId(const PlacementDevice &device)
    {
        switch (device.type)
        {
        case PlacementDevice::CPU:
            return DeviceId::cpu();
        case PlacementDevice::GPU:
            return DeviceId::cuda(device.gpu_index);
        case PlacementDevice::ANY_GPU:
            return DeviceId::cuda(0); // Default to first GPU
        case PlacementDevice::REPLICATED:
            return DeviceId::cpu(); // Replicated tensors loaded to CPU by default
        }
        return DeviceId::cpu();
    }

    /**
     * @brief Convert PlacementDevice to device index (DEPRECATED)
     * @param device PlacementDevice struct
     * @return Device index (0 = CPU, 1+ = GPUs)
     * @deprecated Use toDeviceId() instead
     */
    [[deprecated("Use toDeviceId() instead")]]
    inline int toDeviceIndex(const PlacementDevice &device)
    {
        switch (device.type)
        {
        case PlacementDevice::CPU:
            return 0;
        case PlacementDevice::GPU:
            return 1 + device.gpu_index;
        case PlacementDevice::ANY_GPU:
            return 1; // Default to first GPU
        case PlacementDevice::REPLICATED:
            return 0;
        }
        return 0;
    }

    /**
     * @brief Placement decision for a single layer's weights/compute
     *
     * Specifies where a layer's compute happens and which rank owns the weights.
     * Supports placing attention and FFN on different devices within the same rank.
     *
     * PHASE-AWARE DESIGN:
     * - `device`, `attention_device`, `ffn_device`: Used for PREFILL (compute-bound, GPUs only)
     * - `decode_devices`: Used for DECODE (bandwidth-bound, includes CPU!)
     *
     * During decode, CPUs are first-class participants because decode is memory-bandwidth-bound.
     * A modern Xeon with 6-8 memory channels provides significant bandwidth that should not be wasted.
     */
    struct LayerPlacement
    {
        int layer_idx = -1;      ///< Layer index (0-based)
        int owner_rank = 0;      ///< MPI rank that owns this layer's weights (for TP sharding)
        PlacementDevice device = ///< Device for compute (PREFILL default)
            PlacementDevice::cpu();

        // Fine-grained placement (optional - if attention/FFN on different devices)
        PlacementDevice attention_device = PlacementDevice::cpu();
        PlacementDevice ffn_device = PlacementDevice::cpu();
        bool split_attention_ffn = false; ///< If true, use separate attention_device/ffn_device

        // =====================================================================
        // Phase-Aware Decode Placement (CPU as first-class participant)
        // =====================================================================

        /// Devices participating in DECODE for this layer (bandwidth-proportional)
        /// If empty, falls back to single-device placement (device/attention_device/ffn_device)
        std::vector<PlacementDevice> decode_devices;

        /// Weight shard fractions per decode device (sum to 1.0)
        /// decode_weight_fractions[i] = fraction of weights handled by decode_devices[i]
        std::vector<float> decode_weight_fractions;

        /// Whether to include CPU in decode execution
        bool cpu_participates_in_decode = false;

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

        /// Get decode devices (returns single-device fallback if decode_devices empty)
        std::vector<DeviceId> getDecodeDevices() const
        {
            if (decode_devices.empty())
            {
                // Fallback to single-device (prefill device)
                return {getAttentionDevice()};
            }
            std::vector<DeviceId> result;
            result.reserve(decode_devices.size());
            for (const auto &d : decode_devices)
            {
                result.push_back(toDeviceId(d));
            }
            return result;
        }

        /// Check if CPU participates in decode for this layer
        bool cpuInDecode() const
        {
            if (cpu_participates_in_decode)
                return true;
            for (const auto &d : decode_devices)
            {
                if (d.isCPU())
                    return true;
            }
            return false;
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
        PlacementDevice embedding_device = PlacementDevice::cpu();
        PlacementDevice lm_head_device = PlacementDevice::cpu();
        PlacementDevice final_norm_device = PlacementDevice::cpu();

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
                if (layer.device.isGPU())
                {
                    return true;
                }
            }
            return global.embedding_device.isGPU() ||
                   global.lm_head_device.isGPU();
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
         * @deprecated Use getLayerPlacement(layer_idx).getAttentionDevice() instead
         */
        [[deprecated("Use getLayerPlacement(layer_idx).getAttentionDevice() instead")]]
        int getAttentionDevice(int layer_idx) const
        {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            return getLayerPlacement(layer_idx).getAttentionDeviceIdx();
#pragma GCC diagnostic pop
        }

        /**
         * @brief Get device index for FFN in a layer (for this rank)
         * @param layer_idx Layer index
         * @return Device index (0 = CPU, 1+ = GPU)
         * @deprecated Use getLayerPlacement(layer_idx).getFFNDevice() instead
         */
        [[deprecated("Use getLayerPlacement(layer_idx).getFFNDevice() instead")]]
        int getFFNDevice(int layer_idx) const
        {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            return getLayerPlacement(layer_idx).getFFNDeviceIdx();
#pragma GCC diagnostic pop
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
