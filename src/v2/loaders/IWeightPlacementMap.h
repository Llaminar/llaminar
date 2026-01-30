/**
 * @file IWeightPlacementMap.h
 * @brief Interface for weight-to-device placement mapping
 *
 * Abstracts weight placement decisions to enable:
 * 1. Unit testing without real device placement
 * 2. Configurable placement scenarios in tests
 * 3. Deterministic testing of weight loading logic
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../backends/DeviceId.h"
#include <memory>
#include <string>
#include <vector>

namespace llaminar2 {

// Forward declaration
struct PlacementPlan;

/**
 * @brief Extended device information for phase-aware weight placement
 *
 * Supports the "Selective Duplication" pattern where:
 * - PREFILL: GPU has 100% weights (compute-bound, needs fast matmul)
 * - DECODE: CPU participates with a shard (~20%) (bandwidth-bound)
 *
 * This enables CPU decode participation without duplicating all weights.
 */
struct WeightDeviceInfo {
    DeviceId prefill_device;                  ///< Primary device for prefill (usually GPU)
    std::vector<DeviceId> decode_devices;     ///< Devices participating in decode
    std::vector<float> decode_fractions;      ///< Fraction of weight each decode device handles (sum to 1.0)
    bool cpu_decode_participation = false;    ///< Whether CPU participates in decode

    /// Default constructor - CPU only, no decode participation
    WeightDeviceInfo()
        : prefill_device(DeviceId::cpu())
        , decode_devices{DeviceId::cpu()}
        , decode_fractions{1.0f}
        , cpu_decode_participation(false) {}

    /// Single-device constructor (same device for prefill and decode)
    explicit WeightDeviceInfo(DeviceId device)
        : prefill_device(device)
        , decode_devices{device}
        , decode_fractions{1.0f}
        , cpu_decode_participation(device.is_cpu()) {}

    /// Check if decode uses multiple devices
    bool isDecodeDistributed() const { return decode_devices.size() > 1; }

    /// Get number of decode participants
    size_t decodeDeviceCount() const { return decode_devices.size(); }
};

/**
 * @brief Abstract interface for weight placement mapping
 *
 * This interface abstracts weight-to-device placement decisions to enable:
 * - Unit testing of weight loading without real device placement
 * - Configurable placement scenarios for testing different layouts
 * - Mocking of placement decisions for failure injection
 *
 * Implementations:
 * - WeightPlacementMap: Real implementation with hierarchical lookup
 * - MockWeightPlacementMap: Test implementation with configurable behavior
 */
class IWeightPlacementMap {
public:
    virtual ~IWeightPlacementMap() = default;

    // =========================================================================
    // Core Lookup Methods
    // =========================================================================

    /**
     * @brief Get the device for a specific weight tensor
     *
     * Lookup priority:
     * 1. Exact tensor name match
     * 2. Layer-based lookup (if layer_idx provided and tensor is in a layer)
     * 3. Pattern-based match
     * 4. Default device
     *
     * @param tensor_name Full tensor name (e.g., "blk.0.attn_q.weight")
     * @param layer_idx Optional layer index for layer-based lookup (-1 = not a layer weight)
     * @return DeviceId where this tensor should be allocated
     */
    virtual DeviceId getDeviceForWeight(const std::string& tensor_name, int layer_idx = -1) const = 0;

    /**
     * @brief Get extended device info for phase-aware weight placement
     *
     * Returns full device info including decode participation.
     * For prefill-only code, use getDeviceForWeight() instead.
     *
     * @param tensor_name Full tensor name (e.g., "blk.0.attn_q.weight")
     * @param layer_idx Optional layer index for layer-based lookup (-1 = not a layer weight)
     * @return WeightDeviceInfo with prefill device and decode participation info
     */
    virtual WeightDeviceInfo getDeviceInfoForWeight(const std::string& tensor_name, int layer_idx = -1) const = 0;

    // =========================================================================
    // Configuration Methods
    // =========================================================================

    /**
     * @brief Explicitly map a specific tensor to a device
     * @param tensor_name Exact tensor name
     * @param device Target device
     */
    virtual void setTensorDevice(const std::string& tensor_name, DeviceId device) = 0;

    /**
     * @brief Map an entire layer to a device (bulk assignment)
     * @param layer_idx Layer index (0-based)
     * @param device Target device
     */
    virtual void setLayerDevice(int layer_idx, DeviceId device) = 0;

    /**
     * @brief Map a range of layers to a device
     * @param start_layer First layer (inclusive)
     * @param end_layer Last layer (inclusive)
     * @param device Target device
     */
    virtual void setLayerRange(int start_layer, int end_layer, DeviceId device) = 0;

    /**
     * @brief Set a pattern-based rule (e.g., all "embed.*" → device 0)
     * @param pattern Regex or simple glob pattern
     * @param device Target device
     */
    virtual void setPatternDevice(const std::string& pattern, DeviceId device) = 0;

    // =========================================================================
    // Query Methods
    // =========================================================================

    /**
     * @brief Get the default device for unmapped weights
     */
    virtual DeviceId defaultDevice() const = 0;

    /**
     * @brief Get total number of explicit tensor mappings
     */
    virtual size_t tensorMappingCount() const = 0;

    /**
     * @brief Get total number of layer mappings
     */
    virtual size_t layerMappingCount() const = 0;

    // =========================================================================
    // Block-Level Convenience Methods
    // =========================================================================

    /**
     * @brief Set device for all attention tensors in a layer
     *
     * Sets device for layer tensors matching patterns:
     * - blk.{layer_idx}.attn_q.weight
     * - blk.{layer_idx}.attn_k.weight
     * - blk.{layer_idx}.attn_v.weight
     * - blk.{layer_idx}.attn_output.weight
     * - blk.{layer_idx}.attn_norm.weight
     *
     * @param layer_idx Layer index
     * @param device Target device
     */
    virtual void setAttentionDevice(int layer_idx, DeviceId device) = 0;

    /**
     * @brief Get device for attention block in a layer
     * @param layer_idx Layer index
     * @return DeviceId where attention tensors reside
     */
    virtual DeviceId getAttentionDevice(int layer_idx) const = 0;

    /**
     * @brief Set device for all FFN tensors in a layer
     *
     * Sets device for layer tensors matching patterns:
     * - blk.{layer_idx}.ffn_gate.weight
     * - blk.{layer_idx}.ffn_up.weight
     * - blk.{layer_idx}.ffn_down.weight
     * - blk.{layer_idx}.ffn_norm.weight
     *
     * @param layer_idx Layer index
     * @param device Target device
     */
    virtual void setFFNDevice(int layer_idx, DeviceId device) = 0;

    /**
     * @brief Get device for FFN block in a layer
     * @param layer_idx Layer index
     * @return DeviceId where FFN tensors reside
     */
    virtual DeviceId getFFNDevice(int layer_idx) const = 0;

    // =========================================================================
    // MoE-Specific Methods
    // =========================================================================

    /**
     * @brief Set device for shared expert weights (MoE models)
     * @param expert_idx Expert index
     * @param device Target device
     */
    virtual void setSharedExpertDevice(int expert_idx, DeviceId device) = 0;

    /**
     * @brief Get device for shared expert
     * @param expert_idx Expert index
     * @return DeviceId, or default if not set
     */
    virtual DeviceId getSharedExpertDevice(int expert_idx) const = 0;

    /**
     * @brief Set device for local expert in specific layer (MoE models)
     * @param layer_idx Layer index
     * @param expert_idx Expert index within layer
     * @param device Target device
     */
    virtual void setLocalExpertDevice(int layer_idx, int expert_idx, DeviceId device) = 0;

    /**
     * @brief Get device for local expert in specific layer
     * @param layer_idx Layer index
     * @param expert_idx Expert index within layer
     * @return DeviceId, or default if not set
     */
    virtual DeviceId getLocalExpertDevice(int layer_idx, int expert_idx) const = 0;

    // =========================================================================
    // PlacementPlan Integration
    // =========================================================================

    /**
     * @brief Apply a PlacementPlan to populate all mappings
     *
     * This is the primary entry point for setting up weight placement.
     * Reads the plan (computed by PlacementStrategy) and populates
     * all the layer/tensor/pattern mappings accordingly.
     *
     * @param plan PlacementPlan computed by PlacementStrategy
     */
    virtual void applyPlan(const PlacementPlan& plan) = 0;

    /**
     * @brief Check if a plan has been applied
     * @return true if applyPlan() has been called
     */
    virtual bool hasPlan() const = 0;

    /**
     * @brief Get the strategy name from the applied plan (for logging)
     * @return Strategy name, or empty string if no plan applied
     */
    virtual const std::string& appliedStrategyName() const = 0;

    /**
     * @brief Clear all mappings (reset to default device only)
     */
    virtual void clear() = 0;
};

} // namespace llaminar2
