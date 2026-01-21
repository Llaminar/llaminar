/**
 * @file LayerPlacementConfig.cpp
 * @brief Implementation of layer-to-device placement configuration
 */

#include "config/LayerPlacementConfig.h"
#include <algorithm>
#include <sstream>
#include <set>

namespace llaminar2
{

    // =============================================================================
    // Constructor
    // =============================================================================

    LayerPlacementConfig::LayerPlacementConfig(std::vector<LayerDeviceAssignment> assignments)
        : assignments_(std::move(assignments))
    {
        // Sort by layer index for consistent ordering
        std::sort(assignments_.begin(), assignments_.end(),
                  [](const LayerDeviceAssignment &a, const LayerDeviceAssignment &b)
                  {
                      return a.layer_index < b.layer_index;
                  });
        buildIndex();
    }

    // =============================================================================
    // Index Building
    // =============================================================================

    void LayerPlacementConfig::buildIndex()
    {
        layer_to_device_.clear();
        for (const auto &assignment : assignments_)
        {
            layer_to_device_[assignment.layer_index] = assignment.device;
        }
    }

    // =============================================================================
    // Lookup Methods
    // =============================================================================

    DeviceId LayerPlacementConfig::deviceForLayer(int layer) const
    {
        auto it = layer_to_device_.find(layer);
        if (it == layer_to_device_.end())
        {
            throw std::out_of_range("Layer " + std::to_string(layer) + " not found in placement config");
        }
        return it->second;
    }

    std::vector<int> LayerPlacementConfig::layersForDevice(DeviceId device) const
    {
        std::vector<int> result;
        for (const auto &assignment : assignments_)
        {
            if (assignment.device == device)
            {
                result.push_back(assignment.layer_index);
            }
        }
        // Already sorted because assignments_ is sorted by layer_index
        return result;
    }

    bool LayerPlacementConfig::hasLayersOnCPU() const
    {
        for (const auto &assignment : assignments_)
        {
            if (assignment.device.is_cpu())
            {
                return true;
            }
        }
        return false;
    }

    bool LayerPlacementConfig::hasLayersOnGPU() const
    {
        for (const auto &assignment : assignments_)
        {
            if (assignment.device.is_gpu())
            {
                return true;
            }
        }
        return false;
    }

    DeviceType LayerPlacementConfig::getDeviceTypeForLayer(int layer_idx) const
    {
        return deviceForLayer(layer_idx).type;
    }

    bool LayerPlacementConfig::isGPULayer(int layer_idx) const
    {
        return deviceForLayer(layer_idx).is_gpu();
    }

    bool LayerPlacementConfig::isCPULayer(int layer_idx) const
    {
        return deviceForLayer(layer_idx).is_cpu();
    }

    std::vector<int> LayerPlacementConfig::getGPULayers() const
    {
        std::vector<int> result;
        for (const auto &assignment : assignments_)
        {
            if (assignment.device.is_gpu())
            {
                result.push_back(assignment.layer_index);
            }
        }
        // Already sorted because assignments_ is sorted by layer_index
        return result;
    }

    std::vector<int> LayerPlacementConfig::getCPULayers() const
    {
        std::vector<int> result;
        for (const auto &assignment : assignments_)
        {
            if (assignment.device.is_cpu())
            {
                result.push_back(assignment.layer_index);
            }
        }
        // Already sorted because assignments_ is sorted by layer_index
        return result;
    }

    // =============================================================================
    // Device Inventory
    // =============================================================================

    std::vector<DeviceId> LayerPlacementConfig::uniqueDevices() const
    {
        std::set<DeviceId> device_set;
        for (const auto &assignment : assignments_)
        {
            device_set.insert(assignment.device);
        }
        return std::vector<DeviceId>(device_set.begin(), device_set.end());
    }

    int LayerPlacementConfig::deviceCount() const
    {
        return static_cast<int>(uniqueDevices().size());
    }

    // =============================================================================
    // Transition Points
    // =============================================================================

    std::vector<LayerPlacementConfig::TransitionPoint> LayerPlacementConfig::transitionPoints() const
    {
        std::vector<TransitionPoint> result;

        if (assignments_.size() < 2)
        {
            return result;
        }

        for (size_t i = 0; i < assignments_.size() - 1; ++i)
        {
            const auto &current = assignments_[i];
            const auto &next = assignments_[i + 1];

            // Check if device changes between consecutive layers
            if (current.device != next.device)
            {
                result.push_back({current.layer_index,
                                  next.layer_index,
                                  current.device,
                                  next.device});
            }
        }

        return result;
    }

    std::vector<int> LayerPlacementConfig::getDomainBoundaries() const
    {
        std::vector<int> result;

        if (assignments_.size() < 2)
        {
            return result;
        }

        for (size_t i = 0; i < assignments_.size() - 1; ++i)
        {
            const auto &current = assignments_[i];
            const auto &next = assignments_[i + 1];

            // A domain boundary is the "to" layer where device changes
            if (current.device != next.device)
            {
                result.push_back(next.layer_index);
            }
        }

        return result;
    }

    // =============================================================================
    // Validation
    // =============================================================================

    bool LayerPlacementConfig::validate(int expected_layer_count) const
    {
        validation_error_.clear();

        // Check for correct number of layers
        if (static_cast<int>(assignments_.size()) != expected_layer_count)
        {
            std::ostringstream oss;
            oss << "Expected " << expected_layer_count << " layers, got " << assignments_.size();
            validation_error_ = oss.str();
            return false;
        }

        // Check for missing layers (gaps)
        std::set<int> layer_indices;
        for (const auto &assignment : assignments_)
        {
            layer_indices.insert(assignment.layer_index);
        }

        for (int i = 0; i < expected_layer_count; ++i)
        {
            if (layer_indices.find(i) == layer_indices.end())
            {
                std::ostringstream oss;
                oss << "Missing layer " << i << " in placement config";
                validation_error_ = oss.str();
                return false;
            }
        }

        // Check for duplicate layers
        if (static_cast<int>(layer_indices.size()) != expected_layer_count)
        {
            std::ostringstream oss;
            oss << "Duplicate layer assignments detected";
            validation_error_ = oss.str();
            return false;
        }

        // Check for invalid device IDs
        for (const auto &assignment : assignments_)
        {
            if (!assignment.device.is_valid())
            {
                std::ostringstream oss;
                oss << "Invalid device for layer " << assignment.layer_index;
                validation_error_ = oss.str();
                return false;
            }
        }

        // Check for out-of-range layer indices
        for (const auto &assignment : assignments_)
        {
            if (assignment.layer_index < 0 || assignment.layer_index >= expected_layer_count)
            {
                std::ostringstream oss;
                oss << "Layer index " << assignment.layer_index << " out of range [0, "
                    << expected_layer_count << ")";
                validation_error_ = oss.str();
                return false;
            }
        }

        return true;
    }

    // =============================================================================
    // Factory Methods
    // =============================================================================

    LayerPlacementConfig LayerPlacementConfig::allOnDevice(DeviceId device, int num_layers)
    {
        std::vector<LayerDeviceAssignment> assignments;
        assignments.reserve(num_layers);

        for (int i = 0; i < num_layers; ++i)
        {
            assignments.push_back({i, device, 0});
        }

        return LayerPlacementConfig(std::move(assignments));
    }

    LayerPlacementConfig LayerPlacementConfig::cpuFirstLayers(int cpu_layers, int total_layers, DeviceId gpu)
    {
        if (cpu_layers < 0 || cpu_layers > total_layers)
        {
            throw std::invalid_argument("cpu_layers must be in range [0, total_layers]");
        }

        std::vector<LayerDeviceAssignment> assignments;
        assignments.reserve(total_layers);

        DeviceId cpu = DeviceId::cpu();

        for (int i = 0; i < total_layers; ++i)
        {
            if (i < cpu_layers)
            {
                assignments.push_back({i, cpu, 0});
            }
            else
            {
                assignments.push_back({i, gpu, 0});
            }
        }

        return LayerPlacementConfig(std::move(assignments));
    }

    LayerPlacementConfig LayerPlacementConfig::cpuLastLayers(int cpu_layers, int total_layers, DeviceId gpu)
    {
        if (cpu_layers < 0 || cpu_layers > total_layers)
        {
            throw std::invalid_argument("cpu_layers must be in range [0, total_layers]");
        }

        std::vector<LayerDeviceAssignment> assignments;
        assignments.reserve(total_layers);

        DeviceId cpu = DeviceId::cpu();
        int gpu_layers = total_layers - cpu_layers;

        for (int i = 0; i < total_layers; ++i)
        {
            if (i < gpu_layers)
            {
                assignments.push_back({i, gpu, 0});
            }
            else
            {
                assignments.push_back({i, cpu, 0});
            }
        }

        return LayerPlacementConfig(std::move(assignments));
    }

    LayerPlacementConfig LayerPlacementConfig::alternating(int total_layers, DeviceId device_a, DeviceId device_b)
    {
        std::vector<LayerDeviceAssignment> assignments;
        assignments.reserve(total_layers);

        for (int i = 0; i < total_layers; ++i)
        {
            DeviceId device = (i % 2 == 0) ? device_a : device_b;
            assignments.push_back({i, device, 0});
        }

        return LayerPlacementConfig(std::move(assignments));
    }

    LayerPlacementConfig LayerPlacementConfig::custom(std::vector<LayerDeviceAssignment> assignments)
    {
        return LayerPlacementConfig(std::move(assignments));
    }

} // namespace llaminar2
