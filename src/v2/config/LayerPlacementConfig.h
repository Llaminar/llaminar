/**
 * @file LayerPlacementConfig.h
 * @brief Configuration for layer-to-device placement within a single rank
 *
 * Enables CPU to participate as a compute device alongside GPUs,
 * with individual layers assigned to specific devices. This supports
 * CPU pipeline stage patterns where some layers run on CPU while
 * others run on GPU.
 *
 * Use cases:
 *   - CPU offload: First/last few layers on CPU, middle on GPU
 *   - Memory-constrained: Offload large layers to CPU
 *   - Hybrid compute: Mix CPU and GPU for optimal utilization
 *
 * Usage:
 *   // All layers on GPU (standard pattern)
 *   auto config = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 28);
 *
 *   // First 4 layers on CPU, rest on GPU (CPU pipeline stage)
 *   auto config = LayerPlacementConfig::cpuFirstLayers(4, 28, DeviceId::cuda(0));
 *
 *   // Last 2 layers on CPU (LM head offload)
 *   auto config = LayerPlacementConfig::cpuLastLayers(2, 28, DeviceId::cuda(0));
 */

#pragma once

#include "backends/DeviceId.h"
#include <vector>
#include <unordered_map>
#include <optional>
#include <string>
#include <stdexcept>

namespace llaminar2
{

    /**
     * @brief Assignment of a single layer to a device
     */
    struct LayerDeviceAssignment
    {
        int layer_index;
        DeviceId device;

        // Optional: execution priority (for scheduling)
        // Higher priority layers are scheduled first when dependencies allow
        int priority = 0;

        bool operator==(const LayerDeviceAssignment &other) const
        {
            return layer_index == other.layer_index &&
                   device == other.device &&
                   priority == other.priority;
        }
    };

    /**
     * @brief Configuration for layer-to-device placement within a single rank
     *
     * Enables CPU to participate as a compute device alongside GPUs,
     * with layers assigned to specific devices. This is distinct from
     * pipeline parallelism (which distributes layers across MPI ranks)
     * as it operates within a single rank.
     *
     * Thread safety: Immutable after construction, safe to share across threads.
     */
    class LayerPlacementConfig
    {
    public:
        // =========================================================================
        // Constructors
        // =========================================================================

        /// Default constructor - empty configuration (for validation tests)
        LayerPlacementConfig() = default;

        /// Construct from explicit assignments
        /// @param assignments Vector of layer-to-device assignments
        /// @throws std::invalid_argument if assignments are invalid (duplicates, etc.)
        explicit LayerPlacementConfig(std::vector<LayerDeviceAssignment> assignments);

        // =========================================================================
        // Accessors
        // =========================================================================

        /// Get all layer assignments
        const std::vector<LayerDeviceAssignment> &assignments() const { return assignments_; }

        /// Number of layers configured
        int numLayers() const { return static_cast<int>(assignments_.size()); }

        // =========================================================================
        // Lookup
        // =========================================================================

        /// Get device for a specific layer
        /// @param layer Layer index (0-based)
        /// @return Device assigned to this layer
        /// @throws std::out_of_range if layer is not in configuration
        DeviceId deviceForLayer(int layer) const;

        /// Get device ID for a specific layer (alias for deviceForLayer)
        /// @param layer_idx Layer index (0-based)
        /// @return DeviceId assigned to this layer
        /// @throws std::out_of_range if layer is not in configuration
        DeviceId getDeviceIdForLayer(int layer_idx) const { return deviceForLayer(layer_idx); }

        /// Get device type for a specific layer
        /// @param layer_idx Layer index (0-based)
        /// @return DeviceType (CPU, CUDA, ROCm, etc.)
        /// @throws std::out_of_range if layer is not in configuration
        DeviceType getDeviceTypeForLayer(int layer_idx) const;

        /// Get all layers assigned to a specific device
        /// @param device Device to query
        /// @return Vector of layer indices assigned to this device (sorted)
        std::vector<int> layersForDevice(DeviceId device) const;

        /// Check if any layers are assigned to CPU
        bool hasLayersOnCPU() const;

        /// Check if any layers are assigned to any GPU (CUDA or ROCm)
        bool hasLayersOnGPU() const;

        /// Check if a specific layer is assigned to a GPU
        /// @param layer_idx Layer index (0-based)
        /// @return true if layer is on CUDA or ROCm device
        /// @throws std::out_of_range if layer is not in configuration
        bool isGPULayer(int layer_idx) const;

        /// Check if a specific layer is assigned to CPU
        /// @param layer_idx Layer index (0-based)
        /// @return true if layer is on CPU
        /// @throws std::out_of_range if layer is not in configuration
        bool isCPULayer(int layer_idx) const;

        /// Get all layer indices assigned to GPUs
        /// @return Vector of layer indices on GPU devices (sorted)
        std::vector<int> getGPULayers() const;

        /// Get all layer indices assigned to CPU
        /// @return Vector of layer indices on CPU (sorted)
        std::vector<int> getCPULayers() const;

        // =========================================================================
        // Device Inventory
        // =========================================================================

        /// Get list of unique devices used in this configuration
        /// @return Vector of unique DeviceIds (sorted by type, then ordinal)
        std::vector<DeviceId> uniqueDevices() const;

        /// Number of unique devices
        int deviceCount() const;

        // =========================================================================
        // Transition Points
        // =========================================================================

        /**
         * @brief A point where data moves between devices
         *
         * Transition points identify where explicit data transfers are needed
         * (e.g., CPU→GPU or GPU→CPU copies).
         */
        struct TransitionPoint
        {
            int from_layer; ///< Layer producing data (on from_device)
            int to_layer;   ///< Layer consuming data (on to_device)
            DeviceId from_device;
            DeviceId to_device;

            bool operator==(const TransitionPoint &other) const
            {
                return from_layer == other.from_layer &&
                       to_layer == other.to_layer &&
                       from_device == other.from_device &&
                       to_device == other.to_device;
            }
        };

        /// Find all transition points where device changes between consecutive layers
        /// @return Vector of transition points (in layer order)
        std::vector<TransitionPoint> transitionPoints() const;

        /// Get indices where domain boundaries occur (device changes between adjacent layers)
        /// Returns the "to" layer indices where a transition occurs.
        /// For example, if layers 0-3 are on CPU and 4-9 are on GPU, returns {4}.
        /// @return Vector of layer indices where device changes from previous layer (sorted)
        std::vector<int> getDomainBoundaries() const;

        // =========================================================================
        // Validation
        // =========================================================================

        /// Validate configuration against expected layer count
        /// @param expected_layer_count Total number of layers in model
        /// @return true if configuration is valid
        bool validate(int expected_layer_count) const;

        /// Get validation error message (empty if valid)
        /// @return Error message describing validation failure
        std::string validationError() const { return validation_error_; }

        // =========================================================================
        // Factory Methods
        // =========================================================================

        /// All layers on a single device (standard pattern)
        /// @param device Target device for all layers
        /// @param num_layers Total number of layers
        static LayerPlacementConfig allOnDevice(DeviceId device, int num_layers);

        /// First N layers on CPU, rest on GPU (CPU as first pipeline stage)
        /// @param cpu_layers Number of layers to place on CPU
        /// @param total_layers Total number of layers
        /// @param gpu GPU device for remaining layers
        static LayerPlacementConfig cpuFirstLayers(int cpu_layers, int total_layers, DeviceId gpu);

        /// Last N layers on CPU, first on GPU (for LM head offload)
        /// @param cpu_layers Number of layers to place on CPU
        /// @param total_layers Total number of layers
        /// @param gpu GPU device for initial layers
        static LayerPlacementConfig cpuLastLayers(int cpu_layers, int total_layers, DeviceId gpu);

        /// Alternating layers between two devices (for experimentation)
        /// @param total_layers Total number of layers
        /// @param device_a First device (even layers)
        /// @param device_b Second device (odd layers)
        static LayerPlacementConfig alternating(int total_layers, DeviceId device_a, DeviceId device_b);

        /// Custom assignment from explicit layer-device pairs
        /// @param assignments Custom assignments
        static LayerPlacementConfig custom(std::vector<LayerDeviceAssignment> assignments);

    private:
        std::vector<LayerDeviceAssignment> assignments_;
        std::unordered_map<int, DeviceId> layer_to_device_;
        mutable std::string validation_error_;

        /// Build the layer→device lookup index
        void buildIndex();
    };

} // namespace llaminar2
