/**
 * @file WeightPlacementMap.h
 * @brief Fine-grained weight-to-device placement mapping
 * @author David Sanftenberg
 * @date 2025-10-24
 */

#pragma once

#include "IWeightPlacementMap.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

    // Note: WeightDeviceInfo is now defined in IWeightPlacementMap.h

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
     *
     * Implements IWeightPlacementMap interface for testability.
     */
    class WeightPlacementMap : public IWeightPlacementMap
    {
    public:
        /**
         * @brief Construct a placement map with a default device
         * @param default_device Default device to use when no specific rule matches
         */
        explicit WeightPlacementMap(DeviceId default_device = DeviceId::cpu());

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
        DeviceId getDeviceForWeight(const std::string &tensor_name, int layer_idx = -1) const override;

        /**
         * @brief Get extended device info for phase-aware weight placement
         *
         * Returns full device info including decode participation.
         * For prefill-only code, use getDeviceForWeight() instead.
         *
         * Lookup priority is same as getDeviceForWeight().
         *
         * @param tensor_name Full tensor name (e.g., "blk.0.attn_q.weight")
         * @param layer_idx Optional layer index for layer-based lookup (-1 = not a layer weight)
         * @return WeightDeviceInfo with prefill device and decode participation info
         */
        WeightDeviceInfo getDeviceInfoForWeight(const std::string &tensor_name, int layer_idx = -1) const override;

        /**
         * @brief Explicitly map a specific tensor to a device
         * @param tensor_name Exact tensor name
         * @param device Target device
         */
        void setTensorDevice(const std::string &tensor_name, DeviceId device) override;

        /**
         * @brief Map an entire layer to a device (bulk assignment)
         * @param layer_idx Layer index (0-based)
         * @param device Target device
         */
        void setLayerDevice(int layer_idx, DeviceId device) override;

        /**
         * @brief Map a range of layers to a device
         * @param start_layer First layer (inclusive)
         * @param end_layer Last layer (inclusive)
         * @param device Target device
         */
        void setLayerRange(int start_layer, int end_layer, DeviceId device) override;

        /**
         * @brief Set a pattern-based rule (e.g., all "embed.*" → device 0)
         * @param pattern Regex or simple glob pattern
         * @param device Target device
         */
        void setPatternDevice(const std::string &pattern, DeviceId device) override;

        /**
         * @brief Get the default device for unmapped weights
         */
        DeviceId defaultDevice() const override { return default_device_; }

        /**
         * @brief Get total number of explicit tensor mappings
         */
        size_t tensorMappingCount() const override { return tensor_to_device_.size(); }

        /**
         * @brief Get total number of layer mappings
         */
        size_t layerMappingCount() const override { return layer_to_device_.size(); }

        // ========== Block-Level Convenience Methods (Phase 2) ==========

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
        void setAttentionDevice(int layer_idx, DeviceId device) override;

        /**
         * @brief Get device for attention block in a layer
         *
         * Returns device for attn_q (assumes all attention tensors co-located)
         *
         * @param layer_idx Layer index
         * @return DeviceId where attention tensors reside
         */
        DeviceId getAttentionDevice(int layer_idx) const override;

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
        void setFFNDevice(int layer_idx, DeviceId device) override;

        /**
         * @brief Get device for FFN block in a layer
         *
         * Returns device for ffn_gate (assumes all FFN tensors co-located)
         *
         * @param layer_idx Layer index
         * @return DeviceId where FFN tensors reside
         */
        DeviceId getFFNDevice(int layer_idx) const override;

        // ========== MoE-Specific Methods (Phase 2) ==========

        /**
         * @brief Set device for shared expert weights (MoE models)
         *
         * Shared experts are reused across multiple layers.
         * Maps tensor name pattern: "shared_expert.{expert_idx}.*"
         *
         * @param expert_idx Expert index
         * @param device Target device
         */
        void setSharedExpertDevice(int expert_idx, DeviceId device) override;

        /**
         * @brief Get device for shared expert
         *
         * @param expert_idx Expert index
         * @return DeviceId, or default if not set
         */
        DeviceId getSharedExpertDevice(int expert_idx) const override;

        /**
         * @brief Set device for local expert in specific layer (MoE models)
         *
         * Local experts are layer-specific.
         * Maps tensor name pattern: "blk.{layer_idx}.expert.{expert_idx}.*"
         *
         * @param layer_idx Layer index
         * @param expert_idx Expert index within layer
         * @param device Target device
         */
        void setLocalExpertDevice(int layer_idx, int expert_idx, DeviceId device) override;

        /**
         * @brief Get device for local expert in specific layer
         *
         * @param layer_idx Layer index
         * @param expert_idx Expert index within layer
         * @return DeviceId, or default if not set
         */
        DeviceId getLocalExpertDevice(int layer_idx, int expert_idx) const override;

        /**
         * @brief Clear all mappings (reset to default device only)
         */
        void clear() override;

        // ========== PlacementPlan Integration ==========

        /**
         * @brief Apply a PlacementPlan to populate all mappings
         *
         * This is the primary entry point for setting up weight placement.
         * It reads the plan (computed by PlacementStrategy) and populates
         * all the layer/tensor/pattern mappings accordingly.
         *
         * After calling applyPlan(), getDeviceForWeight() will return
         * the correct device for any weight tensor based on the plan.
         *
         * @param plan PlacementPlan computed by PlacementStrategy
         */
        void applyPlan(const PlacementPlan &plan) override;

        /**
         * @brief Check if a plan has been applied
         * @return true if applyPlan() has been called
         */
        bool hasPlan() const override { return plan_applied_; }

        /**
         * @brief Get the strategy name from the applied plan (for logging)
         * @return Strategy name, or empty string if no plan applied
         */
        const std::string &appliedStrategyName() const override { return applied_strategy_name_; }

    private:
        /**
         * @brief Try to extract layer index from tensor name
         * @param tensor_name Tensor name like "blk.5.attn_q.weight"
         * @return Layer index or -1 if not a layer-specific tensor
         */
        int extractLayerIndex(const std::string &tensor_name) const;

        /**
         * @brief Check if tensor name matches a pattern
         * @param tensor_name Tensor name to check
         * @param pattern Pattern to match against
         * @return true if matches
         */
        bool matchesPattern(const std::string &tensor_name, const std::string &pattern) const;

        DeviceId default_device_;                                     ///< Fallback device
        std::unordered_map<std::string, DeviceId> tensor_to_device_;  ///< Explicit tensor → device
        std::vector<DeviceId> layer_to_device_;                       ///< Layer index → device
        std::unordered_map<std::string, DeviceId> pattern_to_device_; ///< Pattern → device

        // MoE-specific mappings (Phase 2)
        std::unordered_map<int, DeviceId> shared_expert_to_device_;        ///< Shared expert index → device
        std::unordered_map<std::string, DeviceId> local_expert_to_device_; ///< "layer_X:expert_Y" → device

        // Phase-aware decode placement (CPU participation)
        std::vector<std::vector<DeviceId>> layer_decode_devices_; ///< Layer index → decode devices
        std::vector<std::vector<float>> layer_decode_fractions_;  ///< Layer index → decode fractions
        std::vector<bool> layer_cpu_participates_;                ///< Layer index → CPU participates in decode

        // PlacementPlan tracking
        bool plan_applied_ = false;         ///< Whether applyPlan() has been called
        std::string applied_strategy_name_; ///< Strategy name from applied plan
    };

} // namespace llaminar2
