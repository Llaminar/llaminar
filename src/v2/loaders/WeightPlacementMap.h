/**
 * @file WeightPlacementMap.h
 * @brief Fine-grained weight-to-device placement mapping
 * @author David Sanftenberg
 * @date 2025-10-24
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2 {

/**
 * @brief Encodes fine-grained decisions about which device should hold each weight tensor.
 * 
 * WeightPlacementMap separates the "what goes where" policy from the execution
 * of weight loading. It supports multiple granularities:
 * 
 * 1. Per-tensor explicit mapping (highest priority)
 * 2. Per-layer bulk mapping (for transformer blocks)
 * 3. Pattern-based rules (e.g., all "ffn.*" tensors)
 * 4. Default device fallback
 * 
 * This enables use cases like:
 * - MoE: Shared experts on GPU, sparse experts on CPU
 * - Offloading: First N layers GPU, rest CPU
 * - Multi-GPU: Layer-wise partitioning across GPUs
 * - Memory-aware: Fit what you can on GPU, rest on CPU
 */
class WeightPlacementMap {
public:
    /**
     * @brief Construct a placement map with a default device
     * @param default_device_idx Device index to use when no specific rule matches
     */
    explicit WeightPlacementMap(int default_device_idx = 0);

    /**
     * @brief Get the device index for a specific weight tensor
     * 
     * Lookup priority:
     * 1. Exact tensor name match
     * 2. Layer-based lookup (if layer_idx provided and tensor is in a layer)
     * 3. Pattern-based match
     * 4. Default device
     * 
     * @param tensor_name Full tensor name (e.g., "blk.0.attn_q.weight")
     * @param layer_idx Optional layer index for layer-based lookup (-1 = not a layer weight)
     * @return Device index where this tensor should be allocated
     */
    int getDeviceForWeight(const std::string& tensor_name, int layer_idx = -1) const;

    /**
     * @brief Explicitly map a specific tensor to a device
     * @param tensor_name Exact tensor name
     * @param device_idx Target device index
     */
    void setTensorDevice(const std::string& tensor_name, int device_idx);

    /**
     * @brief Map an entire layer to a device (bulk assignment)
     * @param layer_idx Layer index (0-based)
     * @param device_idx Target device index
     */
    void setLayerDevice(int layer_idx, int device_idx);

    /**
     * @brief Map a range of layers to a device
     * @param start_layer First layer (inclusive)
     * @param end_layer Last layer (inclusive)
     * @param device_idx Target device index
     */
    void setLayerRange(int start_layer, int end_layer, int device_idx);

    /**
     * @brief Set a pattern-based rule (e.g., all "embed.*" → device 0)
     * @param pattern Regex or simple glob pattern
     * @param device_idx Target device index
     */
    void setPatternDevice(const std::string& pattern, int device_idx);

    /**
     * @brief Get the default device for unmapped weights
     */
    int defaultDevice() const { return default_device_idx_; }

    /**
     * @brief Get total number of explicit tensor mappings
     */
    size_t tensorMappingCount() const { return tensor_to_device_.size(); }

    /**
     * @brief Get total number of layer mappings
     */
    size_t layerMappingCount() const { return layer_to_device_.size(); }

    /**
     * @brief Clear all mappings (reset to default device only)
     */
    void clear();

private:
    /**
     * @brief Try to extract layer index from tensor name
     * @param tensor_name Tensor name like "blk.5.attn_q.weight"
     * @return Layer index or -1 if not a layer-specific tensor
     */
    int extractLayerIndex(const std::string& tensor_name) const;

    /**
     * @brief Check if tensor name matches a pattern
     * @param tensor_name Tensor name to check
     * @param pattern Pattern to match against
     * @return true if matches
     */
    bool matchesPattern(const std::string& tensor_name, const std::string& pattern) const;

    int default_device_idx_;                                    ///< Fallback device
    std::unordered_map<std::string, int> tensor_to_device_;    ///< Explicit tensor → device
    std::vector<int> layer_to_device_;                         ///< Layer index → device
    std::unordered_map<std::string, int> pattern_to_device_;   ///< Pattern → device
};

} // namespace llaminar2
