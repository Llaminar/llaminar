/**
 * @file WeightPlacementMap.cpp
 * @brief Implementation of fine-grained weight placement mapping
 * @author David Sanftenberg
 * @date 2025-10-24
 */

#include "WeightPlacementMap.h"
#include "../execution/mpi_orchestration/PlacementPlan.h"
#include <regex>
#include <sstream>

namespace llaminar2
{

    WeightPlacementMap::WeightPlacementMap(DeviceId default_device)
        : default_device_(default_device) {}

    DeviceId WeightPlacementMap::getDeviceForWeight(const std::string &tensor_name, int layer_idx) const
    {
        // Priority 1: Exact tensor name match
        auto tensor_it = tensor_to_device_.find(tensor_name);
        if (tensor_it != tensor_to_device_.end())
        {
            return tensor_it->second;
        }

        // Priority 2: Layer-based lookup
        if (layer_idx >= 0 && layer_idx < static_cast<int>(layer_to_device_.size()))
        {
            return layer_to_device_[layer_idx];
        }

        // If layer_idx wasn't provided, try to extract from tensor name
        if (layer_idx < 0)
        {
            int extracted_layer = extractLayerIndex(tensor_name);
            if (extracted_layer >= 0 && extracted_layer < static_cast<int>(layer_to_device_.size()))
            {
                return layer_to_device_[extracted_layer];
            }
        }

        // Priority 3: Pattern-based match
        for (const auto &[pattern, device] : pattern_to_device_)
        {
            if (matchesPattern(tensor_name, pattern))
            {
                return device;
            }
        }

        // Priority 4: Default device
        return default_device_;
    }

    WeightDeviceInfo WeightPlacementMap::getDeviceInfoForWeight(const std::string &tensor_name, int layer_idx) const
    {
        // Get the prefill device using existing logic
        DeviceId prefill_device = getDeviceForWeight(tensor_name, layer_idx);

        // Determine effective layer index
        int effective_layer = layer_idx;
        if (effective_layer < 0)
        {
            effective_layer = extractLayerIndex(tensor_name);
        }

        // Check if we have decode placement info for this layer
        if (effective_layer >= 0 && effective_layer < static_cast<int>(layer_decode_devices_.size()))
        {
            const auto &decode_devices = layer_decode_devices_[effective_layer];
            const auto &decode_fractions = layer_decode_fractions_[effective_layer];
            bool cpu_participates = layer_cpu_participates_[effective_layer];

            if (!decode_devices.empty())
            {
                WeightDeviceInfo info;
                info.prefill_device = prefill_device;
                info.decode_devices = decode_devices;
                info.decode_fractions = decode_fractions;
                info.cpu_decode_participation = cpu_participates;
                return info;
            }
        }

        // Fallback: single device for both prefill and decode
        return WeightDeviceInfo(prefill_device);
    }

    void WeightPlacementMap::setTensorDevice(const std::string &tensor_name, DeviceId device)
    {
        tensor_to_device_[tensor_name] = device;
    }

    void WeightPlacementMap::setLayerDevice(int layer_idx, DeviceId device)
    {
        if (layer_idx >= static_cast<int>(layer_to_device_.size()))
        {
            layer_to_device_.resize(layer_idx + 1, default_device_);
        }
        layer_to_device_[layer_idx] = device;
    }

    void WeightPlacementMap::setLayerRange(int start_layer, int end_layer, DeviceId device)
    {
        for (int layer = start_layer; layer <= end_layer; ++layer)
        {
            setLayerDevice(layer, device);
        }
    }

    void WeightPlacementMap::setPatternDevice(const std::string &pattern, DeviceId device)
    {
        pattern_to_device_[pattern] = device;
    }

    void WeightPlacementMap::clear()
    {
        tensor_to_device_.clear();
        layer_to_device_.clear();
        pattern_to_device_.clear();
        shared_expert_to_device_.clear();
        local_expert_to_device_.clear();
        // Clear decode placement info
        layer_decode_devices_.clear();
        layer_decode_fractions_.clear();
        layer_cpu_participates_.clear();
    }

    // ========== Block-Level Convenience Methods (Phase 2) ==========

    void WeightPlacementMap::setAttentionDevice(int layer_idx, DeviceId device)
    {
        // Set all attention tensors for this layer
        std::string base = "blk." + std::to_string(layer_idx) + ".";
        setTensorDevice(base + "attn_q.weight", device);
        setTensorDevice(base + "attn_k.weight", device);
        setTensorDevice(base + "attn_v.weight", device);
        setTensorDevice(base + "attn_output.weight", device);
        setTensorDevice(base + "attn_norm.weight", device);
    }

    DeviceId WeightPlacementMap::getAttentionDevice(int layer_idx) const
    {
        std::string attn_q_name = "blk." + std::to_string(layer_idx) + ".attn_q.weight";
        return getDeviceForWeight(attn_q_name, layer_idx);
    }

    void WeightPlacementMap::setFFNDevice(int layer_idx, DeviceId device)
    {
        // Set all FFN tensors for this layer
        std::string base = "blk." + std::to_string(layer_idx) + ".";
        setTensorDevice(base + "ffn_gate.weight", device);
        setTensorDevice(base + "ffn_up.weight", device);
        setTensorDevice(base + "ffn_down.weight", device);
        setTensorDevice(base + "ffn_norm.weight", device);
    }

    DeviceId WeightPlacementMap::getFFNDevice(int layer_idx) const
    {
        std::string ffn_gate_name = "blk." + std::to_string(layer_idx) + ".ffn_gate.weight";
        return getDeviceForWeight(ffn_gate_name, layer_idx);
    }

    // ========== MoE-Specific Methods (Phase 2) ==========

    void WeightPlacementMap::setSharedExpertDevice(int expert_idx, DeviceId device)
    {
        shared_expert_to_device_[expert_idx] = device;

        // Also set pattern for all tensors matching this expert
        std::string pattern = "shared_expert." + std::to_string(expert_idx) + ".*";
        setPatternDevice(pattern, device);
    }

    DeviceId WeightPlacementMap::getSharedExpertDevice(int expert_idx) const
    {
        auto it = shared_expert_to_device_.find(expert_idx);
        return (it != shared_expert_to_device_.end()) ? it->second : default_device_;
    }

    void WeightPlacementMap::setLocalExpertDevice(int layer_idx, int expert_idx, DeviceId device)
    {
        std::string key = "layer_" + std::to_string(layer_idx) + ":expert_" + std::to_string(expert_idx);
        local_expert_to_device_[key] = device;

        // Also set pattern for all tensors matching this expert
        std::string pattern = "blk." + std::to_string(layer_idx) + ".expert." + std::to_string(expert_idx) + ".*";
        setPatternDevice(pattern, device);
    }

    DeviceId WeightPlacementMap::getLocalExpertDevice(int layer_idx, int expert_idx) const
    {
        std::string key = "layer_" + std::to_string(layer_idx) + ":expert_" + std::to_string(expert_idx);
        auto it = local_expert_to_device_.find(key);
        return (it != local_expert_to_device_.end()) ? it->second : default_device_;
    }

    int WeightPlacementMap::extractLayerIndex(const std::string &tensor_name) const
    {
        // Match patterns like "blk.5.attn_q.weight" or "model.layers.12.mlp.weight"
        std::regex layer_regex(R"((?:blk|layers?)\.(\d+)\.)");
        std::smatch match;

        if (std::regex_search(tensor_name, match, layer_regex) && match.size() > 1)
        {
            try
            {
                return std::stoi(match[1].str());
            }
            catch (...)
            {
                return -1;
            }
        }

        return -1;
    }

    bool WeightPlacementMap::matchesPattern(const std::string &tensor_name,
                                            const std::string &pattern) const
    {
        // Simple glob-style matching (* wildcard)
        // For Phase 1, support basic prefix/suffix/contains patterns

        // Exact match
        if (pattern == tensor_name)
        {
            return true;
        }

        // Wildcard patterns
        if (pattern.front() == '*' && pattern.back() == '*')
        {
            // *substr* - contains
            std::string substr = pattern.substr(1, pattern.size() - 2);
            return tensor_name.find(substr) != std::string::npos;
        }
        else if (pattern.front() == '*')
        {
            // *suffix - ends with
            std::string suffix = pattern.substr(1);
            return tensor_name.size() >= suffix.size() &&
                   tensor_name.compare(tensor_name.size() - suffix.size(), suffix.size(), suffix) == 0;
        }
        else if (pattern.back() == '*')
        {
            // prefix* - starts with
            std::string prefix = pattern.substr(0, pattern.size() - 1);
            return tensor_name.compare(0, prefix.size(), prefix) == 0;
        }

        // Try regex as fallback (for more complex patterns)
        try
        {
            std::regex regex_pattern(pattern);
            return std::regex_search(tensor_name, regex_pattern);
        }
        catch (...)
        {
            // Invalid regex, no match
            return false;
        }
    }

    // ========== PlacementPlan Integration ==========

    void WeightPlacementMap::applyPlan(const PlacementPlan &plan)
    {
        // Clear existing mappings
        clear();

        // Track that we've applied a plan
        plan_applied_ = true;
        applied_strategy_name_ = plan.strategy_name;

        // Apply global tensor placements
        // Embedding tensor
        setPatternDevice("token_embd*", toDeviceId(plan.global.embedding_device));
        setPatternDevice("embed*", toDeviceId(plan.global.embedding_device));

        // LM head tensor
        setPatternDevice("output*", toDeviceId(plan.global.lm_head_device));
        setPatternDevice("lm_head*", toDeviceId(plan.global.lm_head_device));

        // Final norm
        setPatternDevice("output_norm*", toDeviceId(plan.global.final_norm_device));
        setPatternDevice("final_norm*", toDeviceId(plan.global.final_norm_device));
        setPatternDevice("norm*", toDeviceId(plan.global.final_norm_device)); // Some models use just "norm"

        // Resize decode placement vectors to accommodate all layers
        if (!plan.layers.empty())
        {
            int max_layer = 0;
            for (const auto &layer : plan.layers)
            {
                if (layer.layer_idx > max_layer)
                {
                    max_layer = layer.layer_idx;
                }
            }
            layer_decode_devices_.resize(max_layer + 1);
            layer_decode_fractions_.resize(max_layer + 1);
            layer_cpu_participates_.resize(max_layer + 1, false);
        }

        // Apply per-layer placements
        for (const auto &layer : plan.layers)
        {
            int layer_idx = layer.layer_idx;
            if (layer_idx < 0)
            {
                continue;
            }

            if (layer.split_attention_ffn)
            {
                // Separate devices for attention and FFN
                setAttentionDevice(layer_idx, layer.getAttentionDevice());
                setFFNDevice(layer_idx, layer.getFFNDevice());
            }
            else
            {
                // Same device for entire layer
                setLayerDevice(layer_idx, toDeviceId(layer.device));
            }

            // Store decode placement info from LayerPlacement
            if (!layer.decode_devices.empty())
            {
                std::vector<DeviceId> decode_device_ids;
                decode_device_ids.reserve(layer.decode_devices.size());
                for (const auto &d : layer.decode_devices)
                {
                    decode_device_ids.push_back(toDeviceId(d));
                }
                layer_decode_devices_[layer_idx] = std::move(decode_device_ids);
                layer_decode_fractions_[layer_idx] = layer.decode_weight_fractions;
                layer_cpu_participates_[layer_idx] = layer.cpu_participates_in_decode;
            }
            else
            {
                // No explicit decode devices - use prefill device for decode too
                DeviceId prefill_device = layer.split_attention_ffn
                                              ? layer.getAttentionDevice() // Use attention device as representative
                                              : toDeviceId(layer.device);
                layer_decode_devices_[layer_idx] = {prefill_device};
                layer_decode_fractions_[layer_idx] = {1.0f};
                layer_cpu_participates_[layer_idx] = prefill_device.is_cpu();
            }
        }
    }

} // namespace llaminar2
