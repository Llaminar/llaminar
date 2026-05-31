#pragma once

#include "backends/DeviceId.h"
#include "execution/config/RuntimeConfig.h"
#include "tensors/TensorLayout.h"

#include <cstddef>

namespace llaminar2
{
    class IKVCache;

    struct PrefixPayloadLayout
    {
        DeviceId device = DeviceId::cpu();
        int block_size = 64;
        int first_layer_index = 0;
        int total_layers = 0;
        int fa_layers = 0;
        int gdn_layers = 0;
        int local_kv_heads = 0;
        int kv_head_start = 0;
        int head_dim = 0;
        ActivationPrecision k_precision = ActivationPrecision::FP32;
        ActivationPrecision v_precision = ActivationPrecision::FP32;
        TensorLayout kv_layout = TensorLayout::KV_POS_HEAD_DIM;
        size_t bytes_per_fa_layer_k = 0;
        size_t bytes_per_fa_layer_v = 0;
        size_t hybrid_host_state_bytes = 0;
        size_t hybrid_device_state_bytes = 0;
        size_t hybrid_state_bytes = 0;
        int mtp_layers = 0;
        int mtp_local_kv_heads = 0;
        int mtp_kv_head_start = 0;
        int mtp_head_dim = 0;
        ActivationPrecision mtp_k_precision = ActivationPrecision::FP32;
        ActivationPrecision mtp_v_precision = ActivationPrecision::FP32;
        TensorLayout mtp_kv_layout = TensorLayout::KV_POS_HEAD_DIM;
        size_t bytes_per_mtp_layer_k = 0;
        size_t bytes_per_mtp_layer_v = 0;
        size_t mtp_kv_bytes = 0;
        size_t terminal_hidden_bytes = 0;
        size_t terminal_logits_bytes = 0;
        bool includes_hybrid_state = false;
        bool includes_mtp_state = false;
        bool includes_terminal_hidden = false;
        bool includes_terminal_logits = false;

        size_t faKVBytes() const;
        size_t mtpKVBytes() const;
        size_t totalBytes() const;
        bool compatiblePayloadShape(const PrefixPayloadLayout &other) const;
    };

    PrefixPayloadLayout buildDensePrefixPayloadLayout(
        const IKVCache &kv_cache,
        DeviceId device,
        int block_size,
        size_t terminal_hidden_bytes = 0,
        size_t terminal_logits_bytes = 0);

} // namespace llaminar2
