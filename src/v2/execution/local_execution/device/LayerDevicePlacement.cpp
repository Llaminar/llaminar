/**
 * @file LayerDevicePlacement.cpp
 * @brief Implementation of LayerDevicePlacement and subclasses
 * @author David Sanftenberg
 * @date January 2026
 */

#include "LayerDevicePlacement.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace llaminar2
{

    // =========================================================================
    // Factory Method
    // =========================================================================

    std::unique_ptr<LayerDevicePlacement> LayerDevicePlacement::fromExecutionPlan(
        const RankExecutionPlan &plan,
        int total_layers,
        int total_heads,
        int total_kv_heads)
    {
        // Determine placement strategy based on plan configuration

        // Case 1: Local TP (multiple devices on this rank)
        if (plan.usesLocalTP())
        {
            return std::make_unique<LocalTPPlacement>(
                plan.local_tp_devices,
                plan.local_tp_weights,
                total_layers,
                total_heads,
                total_kv_heads);
        }

        // Case 2: Pipeline parallelism (layer range subset)
        if (plan.usesPipelineParallel())
        {
            // Calculate local heads based on weight shard
            int local_heads = total_heads;
            int local_kv_heads = total_kv_heads;

            if (plan.weight_shard.isSharded())
            {
                // Proportional TP: use work_fraction
                local_heads = static_cast<int>(std::round(
                    total_heads * plan.weight_shard.work_fraction));
                local_kv_heads = static_cast<int>(std::round(
                    total_kv_heads * plan.weight_shard.work_fraction));

                // Ensure at least 1 head
                local_heads = std::max(1, local_heads);
                local_kv_heads = std::max(1, local_kv_heads);
            }

            return std::make_unique<PipelineParallelPlacement>(
                plan.primary_device,
                plan.first_layer,
                plan.last_layer,
                total_layers,
                local_heads,
                local_kv_heads,
                plan.weight_shard.shard_index,
                plan.weight_shard.total_shards,
                plan.has_embedding,
                plan.has_lm_head);
        }

        // Case 3: Single device (simplest case)
        return std::make_unique<SingleDevicePlacement>(
            plan.primary_device,
            total_layers,
            total_heads,
            total_kv_heads);
    }

    // =========================================================================
    // SingleDevicePlacement
    // =========================================================================

    SingleDevicePlacement::SingleDevicePlacement(
        GlobalDeviceAddress device,
        int total_layers,
        int total_heads,
        int total_kv_heads)
        : device_(std::move(device)), total_layers_(total_layers), total_heads_(total_heads), total_kv_heads_(total_kv_heads)
    {
    }

    GlobalDeviceAddress SingleDevicePlacement::deviceForLayer(int /*layer_idx*/) const
    {
        return device_;
    }

    GlobalDeviceAddress SingleDevicePlacement::deviceForStage(const std::string & /*stage_name*/) const
    {
        return device_;
    }

    int SingleDevicePlacement::headsForDevice(const GlobalDeviceAddress & /*device*/) const
    {
        return total_heads_;
    }

    int SingleDevicePlacement::kvHeadsForDevice(const GlobalDeviceAddress & /*device*/) const
    {
        return total_kv_heads_;
    }

    bool SingleDevicePlacement::shouldBuildLayer(int layer_idx) const
    {
        return layer_idx >= 0 && layer_idx < total_layers_;
    }

    int SingleDevicePlacement::shardIndexForDevice(const GlobalDeviceAddress & /*device*/) const
    {
        return 0;
    }

    int SingleDevicePlacement::totalShards() const
    {
        return 1;
    }

    std::vector<GlobalDeviceAddress> SingleDevicePlacement::allDevices() const
    {
        return {device_};
    }

    GlobalDeviceAddress SingleDevicePlacement::primaryDevice() const
    {
        return device_;
    }

    // =========================================================================
    // HybridCpuGpuPlacement
    // =========================================================================

    HybridCpuGpuPlacement::HybridCpuGpuPlacement(
        GlobalDeviceAddress gpu_device,
        GlobalDeviceAddress cpu_device,
        int gpu_layers,
        bool cpu_layers_first,
        int total_layers,
        int total_heads,
        int total_kv_heads)
        : gpu_device_(std::move(gpu_device)), cpu_device_(std::move(cpu_device)), gpu_layers_(gpu_layers), cpu_layers_first_(cpu_layers_first), total_layers_(total_layers), total_heads_(total_heads), total_kv_heads_(total_kv_heads)
    {
        // Validate
        if (gpu_layers < 0 || gpu_layers > total_layers)
        {
            throw std::invalid_argument(
                "gpu_layers must be in [0, total_layers]");
        }
    }

    bool HybridCpuGpuPlacement::isGpuLayer(int layer_idx) const
    {
        if (cpu_layers_first_)
        {
            // CPU: [0, total_layers - gpu_layers)
            // GPU: [total_layers - gpu_layers, total_layers)
            return layer_idx >= (total_layers_ - gpu_layers_);
        }
        else
        {
            // GPU: [0, gpu_layers)
            // CPU: [gpu_layers, total_layers)
            return layer_idx < gpu_layers_;
        }
    }

    GlobalDeviceAddress HybridCpuGpuPlacement::deviceForLayer(int layer_idx) const
    {
        return isGpuLayer(layer_idx) ? gpu_device_ : cpu_device_;
    }

    GlobalDeviceAddress HybridCpuGpuPlacement::deviceForStage(const std::string &stage_name) const
    {
        // Embedding and final stages on GPU if GPU has any layers
        if (gpu_layers_ > 0)
        {
            if (stage_name == "embedding")
            {
                // Embedding near first layers
                return cpu_layers_first_ ? cpu_device_ : gpu_device_;
            }
            if (stage_name == "final_norm" || stage_name == "lm_head")
            {
                // LM head near last layers
                return cpu_layers_first_ ? gpu_device_ : cpu_device_;
            }
        }
        return gpu_device_;
    }

    int HybridCpuGpuPlacement::headsForDevice(const GlobalDeviceAddress & /*device*/) const
    {
        // No TP sharding in hybrid mode - all heads on each device
        return total_heads_;
    }

    int HybridCpuGpuPlacement::kvHeadsForDevice(const GlobalDeviceAddress & /*device*/) const
    {
        return total_kv_heads_;
    }

    bool HybridCpuGpuPlacement::shouldBuildLayer(int layer_idx) const
    {
        return layer_idx >= 0 && layer_idx < total_layers_;
    }

    int HybridCpuGpuPlacement::shardIndexForDevice(const GlobalDeviceAddress & /*device*/) const
    {
        return 0;
    }

    int HybridCpuGpuPlacement::totalShards() const
    {
        return 1;
    }

    std::vector<GlobalDeviceAddress> HybridCpuGpuPlacement::allDevices() const
    {
        std::vector<GlobalDeviceAddress> devices;
        if (gpu_layers_ > 0)
        {
            devices.push_back(gpu_device_);
        }
        if (gpu_layers_ < total_layers_)
        {
            devices.push_back(cpu_device_);
        }
        return devices;
    }

    GlobalDeviceAddress HybridCpuGpuPlacement::primaryDevice() const
    {
        // GPU is primary if it has any layers
        return gpu_layers_ > 0 ? gpu_device_ : cpu_device_;
    }

    // =========================================================================
    // PipelineParallelPlacement
    // =========================================================================

    PipelineParallelPlacement::PipelineParallelPlacement(
        GlobalDeviceAddress device,
        int first_layer,
        int last_layer,
        int total_layers,
        int local_heads,
        int local_kv_heads,
        int shard_index,
        int total_shards,
        bool has_embedding,
        bool has_lm_head)
        : device_(std::move(device)), first_layer_(first_layer), last_layer_(last_layer), total_layers_(total_layers), local_heads_(local_heads), local_kv_heads_(local_kv_heads), shard_index_(shard_index), total_shards_(total_shards), has_embedding_(has_embedding), has_lm_head_(has_lm_head)
    {
    }

    GlobalDeviceAddress PipelineParallelPlacement::deviceForLayer(int /*layer_idx*/) const
    {
        return device_;
    }

    GlobalDeviceAddress PipelineParallelPlacement::deviceForStage(const std::string & /*stage_name*/) const
    {
        return device_;
    }

    int PipelineParallelPlacement::headsForDevice(const GlobalDeviceAddress & /*device*/) const
    {
        return local_heads_;
    }

    int PipelineParallelPlacement::kvHeadsForDevice(const GlobalDeviceAddress & /*device*/) const
    {
        return local_kv_heads_;
    }

    bool PipelineParallelPlacement::shouldBuildLayer(int layer_idx) const
    {
        return layer_idx >= first_layer_ && layer_idx <= last_layer_;
    }

    bool PipelineParallelPlacement::shouldBuildEmbedding() const
    {
        return has_embedding_;
    }

    bool PipelineParallelPlacement::shouldBuildLMHead(int /*total_layers*/) const
    {
        return has_lm_head_;
    }

    int PipelineParallelPlacement::shardIndexForDevice(const GlobalDeviceAddress & /*device*/) const
    {
        return shard_index_;
    }

    int PipelineParallelPlacement::totalShards() const
    {
        return total_shards_;
    }

    std::vector<GlobalDeviceAddress> PipelineParallelPlacement::allDevices() const
    {
        return {device_};
    }

    GlobalDeviceAddress PipelineParallelPlacement::primaryDevice() const
    {
        return device_;
    }

    // =========================================================================
    // LocalTPPlacement
    // =========================================================================

    LocalTPPlacement::LocalTPPlacement(
        std::vector<GlobalDeviceAddress> devices,
        std::vector<float> weights,
        int total_layers,
        int total_heads,
        int total_kv_heads)
        : devices_(std::move(devices)), weights_(std::move(weights)), total_layers_(total_layers), total_heads_(total_heads), total_kv_heads_(total_kv_heads)
    {
        if (devices_.empty())
        {
            throw std::invalid_argument("LocalTPPlacement requires at least one device");
        }

        // Build device-to-index map
        for (size_t i = 0; i < devices_.size(); ++i)
        {
            device_to_index_[devices_[i]] = static_cast<int>(i);
        }

        // Normalize weights if provided, or set equal weights
        if (weights_.empty())
        {
            weights_.resize(devices_.size(), 1.0f / devices_.size());
        }
        else if (weights_.size() != devices_.size())
        {
            throw std::invalid_argument(
                "weights size must match devices size");
        }

        computeHeadDistribution();
    }

    void LocalTPPlacement::computeHeadDistribution()
    {
        head_distribution_.resize(devices_.size());
        kv_head_distribution_.resize(devices_.size());

        // Distribute heads proportionally to weights
        float total_weight = 0.0f;
        for (float w : weights_)
        {
            total_weight += w;
        }

        // First pass: compute proportional distribution
        int remaining_heads = total_heads_;
        int remaining_kv_heads = total_kv_heads_;

        for (size_t i = 0; i < devices_.size(); ++i)
        {
            float fraction = weights_[i] / total_weight;

            // Compute heads for this device
            if (i == devices_.size() - 1)
            {
                // Last device gets remainder
                head_distribution_[i] = remaining_heads;
                kv_head_distribution_[i] = remaining_kv_heads;
            }
            else
            {
                head_distribution_[i] = static_cast<int>(
                    std::round(total_heads_ * fraction));
                kv_head_distribution_[i] = static_cast<int>(
                    std::round(total_kv_heads_ * fraction));

                // Ensure at least 1 head per device
                head_distribution_[i] = std::max(1, head_distribution_[i]);
                kv_head_distribution_[i] = std::max(1, kv_head_distribution_[i]);

                remaining_heads -= head_distribution_[i];
                remaining_kv_heads -= kv_head_distribution_[i];
            }
        }

        // Ensure no negative values in case of rounding errors
        for (size_t i = 0; i < devices_.size(); ++i)
        {
            head_distribution_[i] = std::max(1, head_distribution_[i]);
            kv_head_distribution_[i] = std::max(1, kv_head_distribution_[i]);
        }
    }

    GlobalDeviceAddress LocalTPPlacement::deviceForLayer(int /*layer_idx*/) const
    {
        // In local TP, all devices work on every layer
        // Return the primary device (first one)
        return devices_.empty() ? GlobalDeviceAddress::cpu() : devices_[0];
    }

    GlobalDeviceAddress LocalTPPlacement::deviceForStage(const std::string & /*stage_name*/) const
    {
        return devices_.empty() ? GlobalDeviceAddress::cpu() : devices_[0];
    }

    int LocalTPPlacement::headsForDevice(const GlobalDeviceAddress &device) const
    {
        auto it = device_to_index_.find(device);
        if (it == device_to_index_.end())
        {
            return total_heads_; // Unknown device, return all
        }
        return head_distribution_[it->second];
    }

    int LocalTPPlacement::kvHeadsForDevice(const GlobalDeviceAddress &device) const
    {
        auto it = device_to_index_.find(device);
        if (it == device_to_index_.end())
        {
            return total_kv_heads_; // Unknown device, return all
        }
        return kv_head_distribution_[it->second];
    }

    bool LocalTPPlacement::shouldBuildLayer(int layer_idx) const
    {
        // In local TP, this rank builds all layers
        return layer_idx >= 0 && layer_idx < total_layers_;
    }

    int LocalTPPlacement::shardIndexForDevice(const GlobalDeviceAddress &device) const
    {
        auto it = device_to_index_.find(device);
        return it != device_to_index_.end() ? it->second : 0;
    }

    int LocalTPPlacement::totalShards() const
    {
        return static_cast<int>(devices_.size());
    }

    std::vector<GlobalDeviceAddress> LocalTPPlacement::allDevices() const
    {
        return devices_;
    }

    GlobalDeviceAddress LocalTPPlacement::primaryDevice() const
    {
        return devices_.empty() ? GlobalDeviceAddress::cpu() : devices_[0];
    }

} // namespace llaminar2
