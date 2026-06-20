/**
 * @file HybridCacheManager.cpp
 * @brief Implementation of unified KV/GDN state manager
 */

#include "HybridCacheManager.h"
#include "../../utils/Logger.h"

namespace llaminar2
{

    HybridCacheManager::HybridCacheManager(Config config, IKVCache *kv_cache)
        : kv_cache_(kv_cache)
    {
        const int n_layers = config.n_layers;
        layer_state_types_.resize(n_layers, LayerStateType::KV_CACHE);
        gdn_layer_map_.resize(n_layers, -1);
        kv_layer_map_.resize(n_layers, -1);

        int kv_count = 0;
        int gdn_count = 0;

        for (int i = 0; i < n_layers; ++i)
        {
            const std::string &type = (i < static_cast<int>(config.layer_types.size()))
                                          ? config.layer_types[i]
                                          : "full_attention";

            if (type == "gdn" || type == "GDN")
            {
                layer_state_types_[i] = LayerStateType::GDN_STATE;
                gdn_layer_map_[i] = gdn_count++;

                GDNLayerState state;
                state.n_heads = config.n_heads;
                state.head_dim = config.head_dim;
                state.conv_kernel_size = config.conv_kernel_size;
                state.initialize();
                gdn_states_.push_back(std::move(state));
            }
            else
            {
                layer_state_types_[i] = LayerStateType::KV_CACHE;
                kv_layer_map_[i] = kv_count++;
            }
        }

        LOG_DEBUG("[HybridCacheManager] Created: " << n_layers << " layers"
                                                  << " (" << kv_count << " KV cache, "
                                                  << gdn_count << " GDN state)");
    }

    LayerStateType HybridCacheManager::getLayerStateType(int layer_idx) const
    {
        if (layer_idx < 0 || layer_idx >= static_cast<int>(layer_state_types_.size()))
            return LayerStateType::NONE;
        return layer_state_types_[layer_idx];
    }

    GDNLayerState *HybridCacheManager::getGDNState(int layer_idx)
    {
        if (layer_idx < 0 || layer_idx >= static_cast<int>(gdn_layer_map_.size()))
            return nullptr;
        const int gdn_idx = gdn_layer_map_[layer_idx];
        if (gdn_idx < 0)
            return nullptr;
        return &gdn_states_[gdn_idx];
    }

    const GDNLayerState *HybridCacheManager::getGDNState(int layer_idx) const
    {
        if (layer_idx < 0 || layer_idx >= static_cast<int>(gdn_layer_map_.size()))
            return nullptr;
        const int gdn_idx = gdn_layer_map_[layer_idx];
        if (gdn_idx < 0)
            return nullptr;
        return &gdn_states_[gdn_idx];
    }

    void HybridCacheManager::reset()
    {
        for (auto &state : gdn_states_)
        {
            state.reset();
        }
        // Note: KV cache reset is handled externally by the caller
    }

    size_t HybridCacheManager::gdnMemoryBytes() const
    {
        size_t total = 0;
        for (const auto &state : gdn_states_)
        {
            total += state.memoryBytes();
        }
        return total;
    }

    int HybridCacheManager::kvCacheLayerCount() const
    {
        int count = 0;
        for (const auto &type : layer_state_types_)
        {
            if (type == LayerStateType::KV_CACHE)
                ++count;
        }
        return count;
    }

    int HybridCacheManager::gdnLayerCount() const
    {
        return static_cast<int>(gdn_states_.size());
    }

    int HybridCacheManager::toKVCacheLayerIdx(int layer_idx) const
    {
        if (layer_idx < 0 || layer_idx >= static_cast<int>(kv_layer_map_.size()))
            return -1;
        return kv_layer_map_[layer_idx];
    }

} // namespace llaminar2
