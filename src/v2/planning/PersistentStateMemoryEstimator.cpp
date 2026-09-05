/**
 * @file PersistentStateMemoryEstimator.cpp
 * @brief Implements exact full-attention, GDN, and MTP state sizing.
 */

#include "planning/PersistentStateMemoryEstimator.h"

#include "execution/mtp/MTPCheckpointPolicy.h"
#include "kernels/HybridGDNStateGeometry.h"
#include "planning/KVCacheMemoryEstimator.h"
#include "planning/ModelMemoryProfile.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace llaminar2
{
    namespace
    {
        enum class LayerStorageKind
        {
            FullAttention,
            GDN,
        };

        [[nodiscard]] std::size_t checkedAdd(
            std::size_t left,
            std::size_t right,
            const char *what)
        {
            if (right > std::numeric_limits<std::size_t>::max() - left)
            {
                throw std::overflow_error(
                    std::string("Persistent-state ") + what +
                    " overflows size_t");
            }
            return left + right;
        }

        bool layerHasTensor(
            const ModelMemoryProfile &profile,
            int layer,
            std::string_view fragment)
        {
            return std::any_of(
                profile.tensors.begin(),
                profile.tensors.end(),
                [&](const TensorSizeInfo &tensor)
                {
                    return tensor.layer_index == layer &&
                           tensor.name.find(fragment) != std::string::npos;
                });
        }

        LayerStorageKind classifyLayer(
            const ModelMemoryProfile &profile,
            int layer)
        {
            if (layerHasTensor(profile, layer, ".attn_qkv.weight"))
                return LayerStorageKind::GDN;
            if (layerHasTensor(profile, layer, ".attn_q.weight") ||
                layerHasTensor(profile, layer, ".attn_k.weight") ||
                layerHasTensor(profile, layer, ".attn_v.weight") ||
                layerHasTensor(profile, layer, ".attn_output.weight"))
            {
                return LayerStorageKind::FullAttention;
            }
            if (profile.full_attention_interval > 0 &&
                (layer + 1) % profile.full_attention_interval != 0)
            {
                return LayerStorageKind::GDN;
            }
            return LayerStorageKind::FullAttention;
        }

    } // namespace

    PersistentStateEstimate PersistentStateMemoryEstimator::estimate(
        const ModelMemoryProfile &profile,
        DeviceId device,
        int batch_size,
        int max_seq_len,
        int main_local_kv_heads,
        int mtp_local_kv_heads,
        int local_query_head_start,
        int local_query_heads,
        int first_layer,
        int last_layer,
        const std::string &kv_precision,
        bool mtp_enabled)
    {
        PersistentStateEstimate result;
        const int main_layer_count =
            std::max(0, profile.n_layers - profile.mtp_layer_count);
        const int first_main_layer =
            std::clamp(first_layer, 0, main_layer_count);
        const int requested_last =
            last_layer >= 0 ? last_layer : main_layer_count - 1;
        const int last_main_layer =
            std::min(requested_last, main_layer_count - 1);

        for (int layer = first_main_layer;
             layer <= last_main_layer;
             ++layer)
        {
            if (classifyLayer(profile, layer) ==
                LayerStorageKind::GDN)
            {
                ++result.main_gdn_layers;
            }
            else
            {
                ++result.main_full_attention_layers;
            }
        }

        if (mtp_enabled)
        {
            for (int layer = main_layer_count;
                 layer < profile.n_layers;
                 ++layer)
            {
                if (classifyLayer(profile, layer) ==
                    LayerStorageKind::GDN)
                {
                    ++result.mtp_gdn_layers;
                }
                else
                {
                    ++result.mtp_full_attention_layers;
                }
            }
        }

        /*
         * Runtime owns committed and shifted KV as distinct cache objects.
         * Price those objects independently: a replicated predictor can own
         * every KV head beside a sharded main cache, and cache-local metadata
         * must follow the geometry of its actual owner.
         */
        result.main_kv_cache_bytes = KVCacheMemoryEstimator::estimate(
            result.main_full_attention_layers,
            batch_size,
            max_seq_len,
            main_local_kv_heads,
            profile.head_dim,
            kv_precision,
            device);
        result.mtp_kv_cache_bytes = KVCacheMemoryEstimator::estimate(
            result.mtp_full_attention_layers,
            batch_size,
            max_seq_len,
            mtp_local_kv_heads,
            profile.head_dim,
            kv_precision,
            device);
        result.kv_cache_bytes = checkedAdd(
            result.main_kv_cache_bytes,
            result.mtp_kv_cache_bytes,
            "main plus shifted KV-cache bytes");

        HybridGDNStateGeometry geometry;
        const int resident_gdn_layers =
            result.main_gdn_layers + result.mtp_gdn_layers;
        if (resident_gdn_layers > 0)
        {
            geometry = HybridGDNStateGeometry::resolve(
                profile.n_heads,
                local_query_head_start,
                local_query_heads,
                profile.gdn_group_count,
                profile.gdn_time_step_rank,
                profile.gdn_state_size,
                profile.gdn_inner_size,
                profile.gdn_conv_kernel_size);
            if (device.is_gpu())
            {
                result.live_recurrent_state_bytes =
                    geometry.deviceArenaBytes(
                        resident_gdn_layers,
                        batch_size);
                result.prefix_hybrid_device_state_bytes =
                    geometry.deviceSerializedPayloadBytes(
                        result.main_gdn_layers);
            }
            else
            {
                result.live_recurrent_state_bytes =
                    geometry.localPayloadBytes(resident_gdn_layers);
            }
        }

        if (mtp_enabled)
        {
            const auto checkpointPayloadBytes =
                [&](int gdn_layers)
                {
                    return device.is_gpu()
                               ? geometry.deviceSerializedPayloadBytes(
                                     gdn_layers)
                               : geometry.localPayloadBytes(gdn_layers);
                };
            const size_t main_payload_bytes =
                checkpointPayloadBytes(result.main_gdn_layers) +
                static_cast<size_t>(std::max(0, profile.d_model)) *
                    sizeof(float);
            const size_t shifted_payload_bytes =
                checkpointPayloadBytes(result.mtp_gdn_layers);
            result.checkpoint_state_bytes =
                kMTPConcurrentLiveCheckpointSets *
                (main_payload_bytes + shifted_payload_bytes);

            if (device.is_gpu())
            {
                constexpr size_t kSequenceWordsPerLayer = 2;
                constexpr size_t kSequenceWordBytes =
                    sizeof(int32_t);
                const size_t main_metadata =
                    static_cast<size_t>(
                        result.main_full_attention_layers) *
                    kSequenceWordsPerLayer *
                    kSequenceWordBytes;
                const size_t shifted_metadata =
                    static_cast<size_t>(
                        result.mtp_full_attention_layers) *
                    kSequenceWordsPerLayer *
                    kSequenceWordBytes;
                result.sequence_metadata_bytes =
                    kMTPConcurrentLiveCheckpointSets *
                        (main_metadata + shifted_metadata) +
                    static_cast<size_t>(std::max(1, batch_size)) *
                        main_metadata;
            }
        }

        return result;
    }
}
