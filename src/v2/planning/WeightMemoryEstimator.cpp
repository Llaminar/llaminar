#include "planning/WeightMemoryEstimator.h"
#include "planning/ModelMemoryProfile.h"

namespace llaminar2
{

    float WeightMemoryEstimator::getNativeBytesPerWeight(const std::string &quant_type)
    {
        // Block quantization formats: bytes_per_block / elements_per_block
        if (quant_type == "Q4_0")
            return 18.0f / 32; // 0.5625
        if (quant_type == "Q4_1")
            return 20.0f / 32; // 0.625
        if (quant_type == "Q5_0")
            return 22.0f / 32; // 0.6875
        if (quant_type == "Q5_1")
            return 24.0f / 32; // 0.75
        if (quant_type == "Q8_0")
            return 34.0f / 32; // 1.0625
        if (quant_type == "Q8_1")
            return 36.0f / 32; // 1.125
        if (quant_type == "Q2_K")
            return 0.3125f; // ~10 bits/weight
        if (quant_type == "Q3_K")
            return 0.4375f;
        if (quant_type == "Q4_K" || quant_type == "Q4_K_M" || quant_type == "Q4_K_S")
            return 0.5625f;
        if (quant_type == "Q5_K" || quant_type == "Q5_K_M" || quant_type == "Q5_K_S")
            return 0.6875f;
        if (quant_type == "Q6_K")
            return 0.8125f;
        if (quant_type == "IQ4_NL")
            return 18.0f / 32; // Same block size as Q4_0
        if (quant_type == "F16" || quant_type == "BF16")
            return 2.0f;
        if (quant_type == "F32")
            return 4.0f;
        // Unknown format — assume worst case FP32
        return 4.0f;
    }

    float WeightMemoryEstimator::getCUDAPackedBytesPerWeight(size_t K)
    {
        // CUDA kernels repack Q8_0/Q4_0 into int8 with separate scale arrays.
        // Per-weight: 1 byte (int8 data) + scale overhead amortized over K.
        // Scale = 1 float per group of 32 elements = 4/32 = 0.125 bytes/element.
        // Total ~1.125 bytes/weight for large K, slightly more for small K.
        if (K == 0)
            return 1.125f;
        float scale_overhead = (4.0f * ((static_cast<float>(K) + 31) / 32)) / static_cast<float>(K);
        return 1.0f + scale_overhead;
    }

    float WeightMemoryEstimator::getGPUPackedBytesPerWeight(const std::string &quant_type, size_t K)
    {
        // Match WeightVRAMPool native-VNNI layout: payload + FP16 scales + optional mins/emins per 32 values.
        if (quant_type == "Q4_0" || quant_type == "IQ4_NL")
            return 18.0f / 32.0f;
        if (quant_type == "Q4_1")
            return 20.0f / 32.0f;
        if (quant_type == "Q5_0")
            return 22.0f / 32.0f;
        if (quant_type == "Q5_1")
            return 24.0f / 32.0f;
        if (quant_type == "Q8_0" || quant_type == "Q8_1")
            return 34.0f / 32.0f;
        if (quant_type == "Q2_K")
            return 16.0f / 32.0f;
        if (quant_type == "Q3_K")
            return 16.0f / 32.0f;
        if (quant_type == "Q4_K" || quant_type == "Q4_K_M" || quant_type == "Q4_K_S")
            return 20.0f / 32.0f;
        if (quant_type == "Q5_K" || quant_type == "Q5_K_M" || quant_type == "Q5_K_S")
            return 24.0f / 32.0f;
        if (quant_type == "Q6_K")
            return 28.0f / 32.0f;
        if (quant_type == "F16" || quant_type == "BF16")
            return 2.0f;
        if (quant_type == "F32")
            return 4.0f;

        return getCUDAPackedBytesPerWeight(K);
    }

    float WeightMemoryEstimator::getCPUPackedBytesPerWeight(const std::string &quant_type)
    {
        // CPU VNNI packing expands quantized weights for efficient VNNI/AVX512 processing.
        // Q8_0 → int8 with block scales, ~1.125 bytes/weight
        // Q4_0 → dequant to int8 for VNNI, ~1.125 bytes/weight (after expansion)
        // IQ4_NL → dequant to int8, ~1.125 bytes/weight
        if (quant_type == "Q8_0" || quant_type == "Q4_0" || quant_type == "IQ4_NL" ||
            quant_type == "Q4_K" || quant_type == "Q4_K_M" || quant_type == "Q4_K_S" ||
            quant_type == "Q5_K" || quant_type == "Q5_K_M" || quant_type == "Q5_K_S" ||
            quant_type == "Q6_K")
        {
            return 1.125f; // int8 packed with scale overhead
        }
        if (quant_type == "F16" || quant_type == "BF16")
            return 2.0f;
        if (quant_type == "F32")
            return 4.0f;
        return 1.125f; // Default: assume int8 packing
    }

    bool WeightMemoryEstimator::isShardedTensor(const std::string &name)
    {
        // Column-parallel: attn_q, attn_k, attn_v, ffn_gate, ffn_up, output (lm_head)
        // Row-parallel: attn_output (Wo), ffn_down
        return name.find("attn_q") != std::string::npos ||
               name.find("attn_k") != std::string::npos ||
               name.find("attn_v") != std::string::npos ||
               name.find("attn_output") != std::string::npos ||
               name.find("ffn_gate") != std::string::npos ||
               name.find("ffn_up") != std::string::npos ||
               name.find("ffn_down") != std::string::npos ||
               name == "output.weight";
    }

    bool WeightMemoryEstimator::isReplicatedTensor(const std::string &name)
    {
        // Norms, embedding, and biases are replicated
        return name.find("norm") != std::string::npos ||
               name.find("token_embd") != std::string::npos ||
               name.find("embed_tokens") != std::string::npos ||
               name.find("bias") != std::string::npos;
    }

    WeightEstimate WeightMemoryEstimator::estimate(
        const ModelMemoryProfile &profile,
        DeviceId device,
        int shard_index,
        int total_shards,
        int first_layer,
        int last_layer)
    {
        if (last_layer < 0)
        {
            last_layer = profile.n_layers - 1;
        }

        WeightEstimate est;

        for (const auto &t : profile.tensors)
        {
            // Filter by PP layer range
            if (t.layer_index >= 0)
            {
                if (t.layer_index < first_layer || t.layer_index > last_layer)
                    continue;
            }
            // Non-layer tensors (embedding, lm_head, final_norm) are included on all PP stages
            // In a real PP setup you'd filter embedding to first stage and lm_head to last,
            // but for estimation this is conservative (slight overcount).

            size_t native = t.native_bytes;

            // TP sharding: divide shardable weights by shard count
            if (total_shards > 1 && isShardedTensor(t.name))
            {
                native = native / static_cast<size_t>(total_shards);
            }
            // Replicated tensors: full copy on each shard (no division)

            est.native_bytes += native;

            // Compute device-specific packed size
            size_t device_size;
            if (device.is_gpu())
            {
                float bytes_per_weight = getGPUPackedBytesPerWeight(t.quant_type, t.K);
                size_t elements = t.elements;
                if (total_shards > 1 && isShardedTensor(t.name))
                {
                    elements = elements / static_cast<size_t>(total_shards);
                }
                device_size = static_cast<size_t>(static_cast<float>(elements) * bytes_per_weight);
            }
            else
            {
                // CPU: VNNI packing
                float bytes_per_weight = getCPUPackedBytesPerWeight(t.quant_type);
                size_t elements = t.elements;
                if (total_shards > 1 && isShardedTensor(t.name))
                {
                    elements = elements / static_cast<size_t>(total_shards);
                }
                device_size = static_cast<size_t>(static_cast<float>(elements) * bytes_per_weight);
            }
            est.device_bytes += device_size;
        }

        return est;
    }

} // namespace llaminar2
