/**
 * @file LayerDevicePlacement.h
 * @brief Layer-to-device mapping abstraction for multi-device graph building
 *
 * LayerDevicePlacement provides the abstraction for determining which device
 * runs each layer/stage in a heterogeneous multi-device setup. This enables:
 * - Single device execution (all layers on one device)
 * - CPU spillover (some layers on GPU, rest on CPU)
 * - Pipeline parallelism (different ranks own different layer ranges)
 * - Local tensor parallelism (multiple devices share each layer)
 *
 * Key Design Principles:
 * - Polymorphic: Different placement strategies via subclasses
 * - Factory method: Create from RankExecutionPlan for easy integration
 * - Testable: Pure virtual interface enables mocking
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../../../backends/GlobalDeviceAddress.h"
#include "../../mpi_orchestration/RankExecutionPlan.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

    // =========================================================================
    // LayerDevicePlacement Interface
    // =========================================================================

    /**
     * @brief Abstract interface for layer-to-device mapping
     *
     * Determines which device runs each layer and stage, enabling
     * multi-device and heterogeneous execution strategies.
     */
    class LayerDevicePlacement
    {
    public:
        virtual ~LayerDevicePlacement() = default;

        // =====================================================================
        // Core Query Methods
        // =====================================================================

        /**
         * @brief Get device for a specific transformer layer
         * @param layer_idx Layer index (0-based)
         * @return GlobalDeviceAddress for the layer
         */
        virtual GlobalDeviceAddress deviceForLayer(int layer_idx) const = 0;

        /**
         * @brief Get device for a specific named stage
         *
         * Named stages are model components outside the layer loop:
         * - "embedding": Token embedding lookup
         * - "final_norm": Final RMS normalization
         * - "lm_head": Language model head projection
         *
         * @param stage_name Stage name
         * @return GlobalDeviceAddress for the stage
         */
        virtual GlobalDeviceAddress deviceForStage(const std::string &stage_name) const = 0;

        // =====================================================================
        // Tensor Parallelism Support
        // =====================================================================

        /**
         * @brief Get local n_heads for a specific device
         *
         * For tensor parallelism, heads are sharded across devices.
         * This returns the number of attention heads assigned to a device.
         *
         * @param device Device address
         * @return Number of query attention heads for this device
         */
        virtual int headsForDevice(const GlobalDeviceAddress &device) const = 0;

        /**
         * @brief Get local n_kv_heads for a specific device
         *
         * For GQA models, KV heads may be fewer than query heads.
         * This returns the number of KV heads assigned to a device.
         *
         * @param device Device address
         * @return Number of KV attention heads for this device
         */
        virtual int kvHeadsForDevice(const GlobalDeviceAddress &device) const = 0;

        /**
         * @brief Get TP shard index for a device
         *
         * For weight loading and head assignment, each device has a
         * shard index within the TP group.
         *
         * @param device Device address
         * @return Shard index (0-based)
         */
        virtual int shardIndexForDevice(const GlobalDeviceAddress &device) const = 0;

        /**
         * @brief Get total number of TP shards
         * @return Total shard count (1 for no TP)
         */
        virtual int totalShards() const = 0;

        // =====================================================================
        // Build Control
        // =====================================================================

        /**
         * @brief Check if this rank should build a specific layer
         *
         * For pipeline parallelism, only certain ranks build certain layers.
         *
         * @param layer_idx Layer index
         * @return true if this rank should build the layer
         */
        virtual bool shouldBuildLayer(int layer_idx) const = 0;

        /**
         * @brief Check if this rank should build the embedding stage
         * @return true if this rank owns embedding
         */
        virtual bool shouldBuildEmbedding() const
        {
            // Default: check if layer 0 should be built
            return shouldBuildLayer(0);
        }

        /**
         * @brief Check if this rank should build the LM head stage
         * @param total_layers Total number of layers in model
         * @return true if this rank owns LM head
         */
        virtual bool shouldBuildLMHead(int total_layers) const
        {
            // Default: check if last layer should be built
            return shouldBuildLayer(total_layers - 1);
        }

        // =====================================================================
        // Query Methods
        // =====================================================================

        /**
         * @brief Get all devices used by this placement
         * @return Vector of unique device addresses
         */
        virtual std::vector<GlobalDeviceAddress> allDevices() const = 0;

        /**
         * @brief Get primary device (for single-device or first device in TP)
         * @return Primary device address
         */
        virtual GlobalDeviceAddress primaryDevice() const = 0;

        // =====================================================================
        // Factory Method
        // =====================================================================

        /**
         * @brief Create LayerDevicePlacement from RankExecutionPlan
         *
         * This factory method creates the appropriate placement subclass
         * based on the execution plan's configuration:
         * - No PP, no local TP → SingleDevicePlacement
         * - PP enabled → PipelineParallelPlacement
         * - Local TP enabled → LocalTPPlacement
         * - CPU spillover → HybridCpuGpuPlacement
         *
         * @param plan Execution plan for this rank
         * @param total_layers Total layers in model
         * @param total_heads Total query attention heads
         * @param total_kv_heads Total KV attention heads
         * @return Unique pointer to placement instance
         */
        static std::unique_ptr<LayerDevicePlacement> fromExecutionPlan(
            const RankExecutionPlan &plan,
            int total_layers,
            int total_heads,
            int total_kv_heads);
    };

    // =========================================================================
    // SingleDevicePlacement
    // =========================================================================

    /**
     * @brief All layers on a single device
     *
     * The simplest placement strategy: every layer runs on one device.
     * Used for single-GPU or single-CPU inference without tensor parallelism.
     */
    class SingleDevicePlacement : public LayerDevicePlacement
    {
    public:
        /**
         * @brief Construct single device placement
         * @param device Target device
         * @param total_layers Total layers in model
         * @param total_heads Total query attention heads
         * @param total_kv_heads Total KV attention heads
         */
        SingleDevicePlacement(
            GlobalDeviceAddress device,
            int total_layers,
            int total_heads,
            int total_kv_heads);

        // LayerDevicePlacement interface
        GlobalDeviceAddress deviceForLayer(int layer_idx) const override;
        GlobalDeviceAddress deviceForStage(const std::string &stage_name) const override;
        int headsForDevice(const GlobalDeviceAddress &device) const override;
        int kvHeadsForDevice(const GlobalDeviceAddress &device) const override;
        bool shouldBuildLayer(int layer_idx) const override;
        int shardIndexForDevice(const GlobalDeviceAddress &device) const override;
        int totalShards() const override;
        std::vector<GlobalDeviceAddress> allDevices() const override;
        GlobalDeviceAddress primaryDevice() const override;

    private:
        GlobalDeviceAddress device_;
        int total_layers_;
        int total_heads_;
        int total_kv_heads_;
    };

    // =========================================================================
    // HybridCpuGpuPlacement
    // =========================================================================

    /**
     * @brief CPU spillover: some layers on GPU, rest on CPU
     *
     * When GPU memory is insufficient for all layers, this placement
     * offloads some layers to CPU. The layers can be arranged:
     * - cpu_layers_first=true: CPU runs first N layers, GPU runs rest
     * - cpu_layers_first=false: GPU runs first N layers, CPU runs rest
     *
     * Typically, GPU layers are "first" (closer to embedding) for better
     * memory access patterns during decode.
     */
    class HybridCpuGpuPlacement : public LayerDevicePlacement
    {
    public:
        /**
         * @brief Construct hybrid CPU/GPU placement
         * @param gpu_device GPU device address
         * @param cpu_device CPU device address
         * @param gpu_layers Number of layers to run on GPU
         * @param cpu_layers_first If true, CPU layers come first (0..N-1)
         * @param total_layers Total layers in model
         * @param total_heads Total query attention heads
         * @param total_kv_heads Total KV attention heads
         */
        HybridCpuGpuPlacement(
            GlobalDeviceAddress gpu_device,
            GlobalDeviceAddress cpu_device,
            int gpu_layers,
            bool cpu_layers_first,
            int total_layers,
            int total_heads,
            int total_kv_heads);

        // LayerDevicePlacement interface
        GlobalDeviceAddress deviceForLayer(int layer_idx) const override;
        GlobalDeviceAddress deviceForStage(const std::string &stage_name) const override;
        int headsForDevice(const GlobalDeviceAddress &device) const override;
        int kvHeadsForDevice(const GlobalDeviceAddress &device) const override;
        bool shouldBuildLayer(int layer_idx) const override;
        int shardIndexForDevice(const GlobalDeviceAddress &device) const override;
        int totalShards() const override;
        std::vector<GlobalDeviceAddress> allDevices() const override;
        GlobalDeviceAddress primaryDevice() const override;

        // Query methods
        bool isGpuLayer(int layer_idx) const;
        int gpuLayerCount() const { return gpu_layers_; }
        int cpuLayerCount() const { return total_layers_ - gpu_layers_; }

    private:
        GlobalDeviceAddress gpu_device_;
        GlobalDeviceAddress cpu_device_;
        int gpu_layers_;
        bool cpu_layers_first_;
        int total_layers_;
        int total_heads_;
        int total_kv_heads_;
    };

    // =========================================================================
    // PipelineParallelPlacement
    // =========================================================================

    /**
     * @brief Pipeline parallelism: each rank owns a range of layers
     *
     * In PP mode, each MPI rank is responsible for a contiguous range
     * of transformer layers. This placement represents a single rank's
     * view of its layer range.
     *
     * Key behaviors:
     * - shouldBuildLayer() returns true only for owned layers
     * - shouldBuildEmbedding() returns true only for first PP stage
     * - shouldBuildLMHead() returns true only for last PP stage
     */
    class PipelineParallelPlacement : public LayerDevicePlacement
    {
    public:
        /**
         * @brief Construct pipeline parallel placement
         * @param device Device for this rank's layers
         * @param first_layer First layer index (inclusive)
         * @param last_layer Last layer index (inclusive)
         * @param total_layers Total layers in model (for bounds checking)
         * @param local_heads Query heads for this rank
         * @param local_kv_heads KV heads for this rank
         * @param shard_index TP shard index (for weight loading)
         * @param total_shards Total TP shards
         * @param has_embedding Whether this rank owns embedding
         * @param has_lm_head Whether this rank owns LM head
         */
        PipelineParallelPlacement(
            GlobalDeviceAddress device,
            int first_layer,
            int last_layer,
            int total_layers,
            int local_heads,
            int local_kv_heads,
            int shard_index,
            int total_shards,
            bool has_embedding = false,
            bool has_lm_head = false);

        // LayerDevicePlacement interface
        GlobalDeviceAddress deviceForLayer(int layer_idx) const override;
        GlobalDeviceAddress deviceForStage(const std::string &stage_name) const override;
        int headsForDevice(const GlobalDeviceAddress &device) const override;
        int kvHeadsForDevice(const GlobalDeviceAddress &device) const override;
        bool shouldBuildLayer(int layer_idx) const override;
        bool shouldBuildEmbedding() const override;
        bool shouldBuildLMHead(int total_layers) const override;
        int shardIndexForDevice(const GlobalDeviceAddress &device) const override;
        int totalShards() const override;
        std::vector<GlobalDeviceAddress> allDevices() const override;
        GlobalDeviceAddress primaryDevice() const override;

        // Query methods
        int firstLayer() const { return first_layer_; }
        int lastLayer() const { return last_layer_; }
        int layerCount() const { return last_layer_ - first_layer_ + 1; }

    private:
        GlobalDeviceAddress device_;
        int first_layer_;
        int last_layer_;
        int total_layers_;
        int local_heads_;
        int local_kv_heads_;
        int shard_index_;
        int total_shards_;
        bool has_embedding_;
        bool has_lm_head_;
    };

    // =========================================================================
    // LocalTPPlacement
    // =========================================================================

    /**
     * @brief Local tensor parallelism: multiple devices share each layer
     *
     * In LOCAL TP mode, multiple devices on the same rank work together
     * on each layer. Heads and FFN dimensions are sharded across devices,
     * with optional weighted distribution for heterogeneous GPUs.
     *
     * Example: NVIDIA 4090 (73%) + AMD 7900XTX (27%) sharing each layer
     */
    class LocalTPPlacement : public LayerDevicePlacement
    {
    public:
        /**
         * @brief Construct local TP placement
         * @param devices Devices participating in TP
         * @param weights Work distribution weights (empty = equal)
         * @param total_layers Total layers in model
         * @param total_heads Total query attention heads
         * @param total_kv_heads Total KV attention heads
         */
        LocalTPPlacement(
            std::vector<GlobalDeviceAddress> devices,
            std::vector<float> weights,
            int total_layers,
            int total_heads,
            int total_kv_heads);

        // LayerDevicePlacement interface
        GlobalDeviceAddress deviceForLayer(int layer_idx) const override;
        GlobalDeviceAddress deviceForStage(const std::string &stage_name) const override;
        int headsForDevice(const GlobalDeviceAddress &device) const override;
        int kvHeadsForDevice(const GlobalDeviceAddress &device) const override;
        bool shouldBuildLayer(int layer_idx) const override;
        int shardIndexForDevice(const GlobalDeviceAddress &device) const override;
        int totalShards() const override;
        std::vector<GlobalDeviceAddress> allDevices() const override;
        GlobalDeviceAddress primaryDevice() const override;

        // Query methods
        const std::vector<float> &weights() const { return weights_; }
        const std::vector<int> &headDistribution() const { return head_distribution_; }
        const std::vector<int> &kvHeadDistribution() const { return kv_head_distribution_; }

    private:
        std::vector<GlobalDeviceAddress> devices_;
        std::vector<float> weights_;
        int total_layers_;
        int total_heads_;
        int total_kv_heads_;
        std::vector<int> head_distribution_;
        std::vector<int> kv_head_distribution_;
        std::unordered_map<GlobalDeviceAddress, int> device_to_index_;

        void computeHeadDistribution();
    };

} // namespace llaminar2
