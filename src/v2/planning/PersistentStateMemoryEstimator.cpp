/**
 * @file PersistentStateMemoryEstimator.cpp
 * @brief Implements exact full-attention, GDN, and MTP state sizing.
 */

#include "planning/PersistentStateMemoryEstimator.h"

#include "execution/mtp/MTPCheckpointPolicy.h"
#include "planning/KVCacheMemoryEstimator.h"
#include "planning/ModelMemoryProfile.h"

#include <algorithm>
#include <cstdint>
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

        struct RecurrentGeometry
        {
            size_t local_conv_floats = 0;
            size_t full_conv_floats = 0;
            size_t local_recurrence_floats = 0;
            size_t full_recurrence_floats = 0;
        };

        size_t alignUp(size_t bytes, size_t alignment)
        {
            return (bytes + alignment - 1) & ~(alignment - 1);
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

        RecurrentGeometry recurrentGeometry(
            const ModelMemoryProfile &profile,
            int local_query_heads,
            int total_shards)
        {
            RecurrentGeometry geometry;
            if (profile.gdn_state_size <= 0 ||
                profile.gdn_conv_kernel_size <= 1 ||
                profile.n_heads <= 0)
            {
                return geometry;
            }

            const int full_k_heads =
                profile.gdn_group_count > 0
                    ? profile.gdn_group_count
                    : profile.n_heads;
            const int full_v_heads =
                profile.gdn_time_step_rank > 0
                    ? profile.gdn_time_step_rank
                    : full_k_heads;
            if (full_k_heads <= 0 || full_v_heads <= 0)
                return geometry;

            const int shards = std::max(1, total_shards);
            const int local_attention_heads =
                local_query_heads > 0
                    ? local_query_heads
                    : std::max(1, profile.n_heads / shards);
            int local_k_heads = full_k_heads;
            int local_v_heads = full_v_heads;
            const bool modular_repeat = full_v_heads > full_k_heads;
            if (local_attention_heads < profile.n_heads)
            {
                local_v_heads = std::max(
                    1,
                    full_v_heads * local_attention_heads /
                        profile.n_heads);
                if (!modular_repeat)
                {
                    local_k_heads = std::max(
                        1,
                        full_k_heads * local_attention_heads /
                            profile.n_heads);
                }
            }

            const size_t state_dim =
                static_cast<size_t>(profile.gdn_state_size);
            const size_t full_value_dim =
                profile.gdn_inner_size > 0
                    ? static_cast<size_t>(profile.gdn_inner_size)
                    : static_cast<size_t>(full_v_heads) * state_dim;
            const size_t local_value_dim =
                profile.gdn_inner_size > 0
                    ? static_cast<size_t>(profile.gdn_inner_size) *
                          static_cast<size_t>(local_v_heads) /
                          static_cast<size_t>(full_v_heads)
                    : static_cast<size_t>(local_v_heads) * state_dim;
            const size_t history =
                static_cast<size_t>(
                    profile.gdn_conv_kernel_size - 1);

            geometry.full_conv_floats =
                (2 * static_cast<size_t>(full_k_heads) * state_dim +
                 full_value_dim) *
                history;
            geometry.local_conv_floats =
                (2 * static_cast<size_t>(local_k_heads) * state_dim +
                 local_value_dim) *
                history;
            geometry.full_recurrence_floats =
                static_cast<size_t>(full_v_heads) *
                state_dim * state_dim;
            geometry.local_recurrence_floats =
                static_cast<size_t>(local_v_heads) *
                state_dim * state_dim;
            return geometry;
        }

        size_t gpuArenaBytesForOneKernel(
            size_t local_floats,
            size_t full_floats,
            int request_capacity)
        {
            if (local_floats == 0)
                return 0;

            constexpr size_t kAlignment = 256;
            constexpr size_t kFP32 = sizeof(float);
            size_t bytes =
                alignUp(local_floats * kFP32, kAlignment);
            if (full_floats != local_floats)
            {
                bytes +=
                    alignUp(full_floats * kFP32, kAlignment);
            }
            bytes += alignUp(
                static_cast<size_t>(std::max(1, request_capacity)) *
                    std::max(local_floats, full_floats) * kFP32,
                kAlignment);
            return bytes;
        }

        size_t localPayloadBytes(
            const RecurrentGeometry &geometry,
            int layer_count)
        {
            return static_cast<size_t>(std::max(0, layer_count)) *
                   (geometry.local_conv_floats +
                    geometry.local_recurrence_floats) *
                   sizeof(float);
        }
    } // namespace

    PersistentStateEstimate PersistentStateMemoryEstimator::estimate(
        const ModelMemoryProfile &profile,
        DeviceId device,
        int batch_size,
        int max_seq_len,
        int local_kv_heads,
        int local_query_heads,
        int total_shards,
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

        const int resident_fa_layers =
            result.main_full_attention_layers +
            result.mtp_full_attention_layers;
        result.kv_cache_bytes = KVCacheMemoryEstimator::estimate(
            resident_fa_layers,
            batch_size,
            max_seq_len,
            local_kv_heads,
            profile.head_dim,
            kv_precision,
            device);

        const RecurrentGeometry geometry =
            recurrentGeometry(profile, local_query_heads, total_shards);
        if (device.is_gpu())
        {
            const size_t bytes_per_gdn_layer =
                gpuArenaBytesForOneKernel(
                    geometry.local_conv_floats,
                    geometry.full_conv_floats,
                    batch_size) +
                gpuArenaBytesForOneKernel(
                    geometry.local_recurrence_floats,
                    geometry.full_recurrence_floats,
                    batch_size);
            result.live_recurrent_state_bytes =
                static_cast<size_t>(
                    result.main_gdn_layers +
                    result.mtp_gdn_layers) *
                bytes_per_gdn_layer;
        }
        else
        {
            result.live_recurrent_state_bytes =
                localPayloadBytes(
                    geometry,
                    result.main_gdn_layers +
                        result.mtp_gdn_layers);
        }

        if (mtp_enabled)
        {
            const size_t main_payload_bytes =
                localPayloadBytes(
                    geometry,
                    result.main_gdn_layers) +
                static_cast<size_t>(std::max(0, profile.d_model)) *
                    sizeof(float);
            const size_t shifted_payload_bytes =
                localPayloadBytes(
                    geometry,
                    result.mtp_gdn_layers);
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
