/**
 * @file WeightPlacementMap.cpp
 * @brief Implementation of fine-grained weight placement mapping
 * @author David Sanftenberg
 * @date 2025-10-24
 */

#include "WeightPlacementMap.h"
#include <regex>
#include <sstream>

namespace llaminar2 {

WeightPlacementMap::WeightPlacementMap(int default_device_idx)
    : default_device_idx_(default_device_idx) {}

int WeightPlacementMap::getDeviceForWeight(const std::string& tensor_name, int layer_idx) const {
    // Priority 1: Exact tensor name match
    auto tensor_it = tensor_to_device_.find(tensor_name);
    if (tensor_it != tensor_to_device_.end()) {
        return tensor_it->second;
    }

    // Priority 2: Layer-based lookup
    if (layer_idx >= 0 && layer_idx < static_cast<int>(layer_to_device_.size())) {
        return layer_to_device_[layer_idx];
    }

    // If layer_idx wasn't provided, try to extract from tensor name
    if (layer_idx < 0) {
        int extracted_layer = extractLayerIndex(tensor_name);
        if (extracted_layer >= 0 && extracted_layer < static_cast<int>(layer_to_device_.size())) {
            return layer_to_device_[extracted_layer];
        }
    }

    // Priority 3: Pattern-based match
    for (const auto& [pattern, device_idx] : pattern_to_device_) {
        if (matchesPattern(tensor_name, pattern)) {
            return device_idx;
        }
    }

    // Priority 4: Default device
    return default_device_idx_;
}

void WeightPlacementMap::setTensorDevice(const std::string& tensor_name, int device_idx) {
    tensor_to_device_[tensor_name] = device_idx;
}

void WeightPlacementMap::setLayerDevice(int layer_idx, int device_idx) {
    if (layer_idx >= static_cast<int>(layer_to_device_.size())) {
        layer_to_device_.resize(layer_idx + 1, default_device_idx_);
    }
    layer_to_device_[layer_idx] = device_idx;
}

void WeightPlacementMap::setLayerRange(int start_layer, int end_layer, int device_idx) {
    for (int layer = start_layer; layer <= end_layer; ++layer) {
        setLayerDevice(layer, device_idx);
    }
}

void WeightPlacementMap::setPatternDevice(const std::string& pattern, int device_idx) {
    pattern_to_device_[pattern] = device_idx;
}

void WeightPlacementMap::clear() {
    tensor_to_device_.clear();
    layer_to_device_.clear();
    pattern_to_device_.clear();
}

int WeightPlacementMap::extractLayerIndex(const std::string& tensor_name) const {
    // Match patterns like "blk.5.attn_q.weight" or "model.layers.12.mlp.weight"
    std::regex layer_regex(R"((?:blk|layers?)\.(\d+)\.)");
    std::smatch match;
    
    if (std::regex_search(tensor_name, match, layer_regex) && match.size() > 1) {
        try {
            return std::stoi(match[1].str());
        } catch (...) {
            return -1;
        }
    }
    
    return -1;
}

bool WeightPlacementMap::matchesPattern(const std::string& tensor_name, 
                                       const std::string& pattern) const {
    // Simple glob-style matching (* wildcard)
    // For Phase 1, support basic prefix/suffix/contains patterns
    
    // Exact match
    if (pattern == tensor_name) {
        return true;
    }
    
    // Wildcard patterns
    if (pattern.front() == '*' && pattern.back() == '*') {
        // *substr* - contains
        std::string substr = pattern.substr(1, pattern.size() - 2);
        return tensor_name.find(substr) != std::string::npos;
    } else if (pattern.front() == '*') {
        // *suffix - ends with
        std::string suffix = pattern.substr(1);
        return tensor_name.size() >= suffix.size() &&
               tensor_name.compare(tensor_name.size() - suffix.size(), suffix.size(), suffix) == 0;
    } else if (pattern.back() == '*') {
        // prefix* - starts with
        std::string prefix = pattern.substr(0, pattern.size() - 1);
        return tensor_name.compare(0, prefix.size(), prefix) == 0;
    }
    
    // Try regex as fallback (for more complex patterns)
    try {
        std::regex regex_pattern(pattern);
        return std::regex_search(tensor_name, regex_pattern);
    } catch (...) {
        // Invalid regex, no match
        return false;
    }
}

} // namespace llaminar2
