#include "planning/WeightMemoryEstimator.h"
#include "planning/ModelMemoryProfile.h"
#include "tensors/BlockStructures.h"

/**
 * @file WeightMemoryEstimator.cpp
 * @brief Estimates native and prepared weight memory across CPU and GPU devices.
 *
 * Native estimates mirror GGUF block layouts from BlockStructures.h. GPU estimates
 * mirror the separated native-VNNI layout used by WeightVRAMPool/CUDA/ROCm packers:
 * per-32-value payload bytes plus FP16 scales and optional FP16 mins/Q2_K emins.
 * CPU estimates intentionally preserve the existing VNNI-expanded planning model.
 */

namespace llaminar2
{
    namespace
    {
        struct NativeBlockLayout
        {
            const char *quant_type;
            size_t block_bytes;
            size_t block_elements;
        };

        struct GPUNativeVNNILayout
        {
            const char *quant_type;
            int payload_bytes_per_32;
            bool is_asymmetric;
            bool has_emins;
        };

        constexpr NativeBlockLayout kNativeBlockLayouts[] = {
            {"Q4_0", sizeof(Q4_0Block), Q4_0Block::BLOCK_SIZE},
            {"Q4_1", sizeof(Q4_1Block), Q4_1Block::BLOCK_SIZE},
            {"Q5_0", sizeof(Q5_0Block), Q5_0Block::BLOCK_SIZE},
            {"Q5_1", sizeof(Q5_1Block), Q5_1Block::BLOCK_SIZE},
            {"Q8_0", sizeof(Q8_0Block), Q8_0Block::BLOCK_SIZE},
            {"Q8_1", sizeof(Q8_1Block), Q8_1Block::BLOCK_SIZE},
            {"Q2_K", sizeof(Q2_KBlock), Q2_KBlock::BLOCK_SIZE},
            {"Q3_K", sizeof(Q3_KBlock), Q3_KBlock::BLOCK_SIZE},
            {"Q3_K_S", sizeof(Q3_KBlock), Q3_KBlock::BLOCK_SIZE},
            {"Q3_K_M", sizeof(Q3_KBlock), Q3_KBlock::BLOCK_SIZE},
            {"Q3_K_L", sizeof(Q3_KBlock), Q3_KBlock::BLOCK_SIZE},
            {"Q4_K", sizeof(Q4_KBlock), Q4_KBlock::BLOCK_SIZE},
            {"Q4_K_S", sizeof(Q4_KBlock), Q4_KBlock::BLOCK_SIZE},
            {"Q4_K_M", sizeof(Q4_KBlock), Q4_KBlock::BLOCK_SIZE},
            {"Q5_K", sizeof(Q5_KBlock), Q5_KBlock::BLOCK_SIZE},
            {"Q5_K_S", sizeof(Q5_KBlock), Q5_KBlock::BLOCK_SIZE},
            {"Q5_K_M", sizeof(Q5_KBlock), Q5_KBlock::BLOCK_SIZE},
            {"Q6_K", sizeof(Q6_KBlock), Q6_KBlock::BLOCK_SIZE},
            {"Q8_K", sizeof(Q8_KBlock), Q8_KBlock::BLOCK_SIZE},
            {"IQ4_NL", sizeof(IQ4_NLBlock), IQ4_NLBlock::BLOCK_SIZE},
            {"IQ4_XS", sizeof(IQ4_XSBlock), IQ4_XSBlock::BLOCK_SIZE},
            {"IQ2_XXS", sizeof(IQ2_XXSBlock), IQ2_XXSBlock::BLOCK_SIZE},
            {"IQ2_XS", sizeof(IQ2_XSBlock), IQ2_XSBlock::BLOCK_SIZE},
            {"IQ3_XXS", sizeof(IQ3_XXSBlock), IQ3_XXSBlock::BLOCK_SIZE},
            {"IQ2_S", sizeof(IQ2_SBlock), IQ2_SBlock::BLOCK_SIZE},
            {"IQ3_S", sizeof(IQ3_SBlock), IQ3_SBlock::BLOCK_SIZE},
            {"IQ1_S", sizeof(IQ1_SBlock), IQ1_SBlock::BLOCK_SIZE},
            {"IQ1_M", sizeof(IQ1_MBlock), IQ1_MBlock::BLOCK_SIZE},
        };

        constexpr GPUNativeVNNILayout kGPUNativeVNNILayouts[] = {
            {"Q4_0", 16, false, false},
            {"IQ4_NL", 16, false, false},
            {"Q4_1", 16, true, false},
            {"Q5_0", 20, false, false},
            {"Q5_1", 20, true, false},
            {"Q8_0", 32, false, false},
            {"Q8_1", 32, false, false},
            {"Q2_K", 8, true, true},
            {"Q3_K", 12, true, false},
            {"Q3_K_S", 12, true, false},
            {"Q3_K_M", 12, true, false},
            {"Q3_K_L", 12, true, false},
            {"Q4_K", 16, true, false},
            {"Q4_K_S", 16, true, false},
            {"Q4_K_M", 16, true, false},
            {"Q5_K", 20, true, false},
            {"Q5_K_S", 20, true, false},
            {"Q5_K_M", 20, true, false},
            {"Q6_K", 24, true, false},
            {"IQ4_XS", 16, false, false},
            {"IQ2_XXS", 8, false, false},
            {"IQ2_XS", 9, true, false},
            {"IQ3_XXS", 12, false, false},
            {"IQ2_S", 9, true, false},
            {"IQ3_S", 13, false, false},
            {"IQ1_S", 6, true, false},
            {"IQ1_M", 6, true, false},
        };

        const NativeBlockLayout *findNativeBlockLayout(const std::string &quant_type)
        {
            for (const auto &layout : kNativeBlockLayouts)
            {
                if (quant_type == layout.quant_type)
                    return &layout;
            }
            return nullptr;
        }

        const GPUNativeVNNILayout *findGPUNativeVNNILayout(const std::string &quant_type)
        {
            for (const auto &layout : kGPUNativeVNNILayouts)
            {
                if (quant_type == layout.quant_type)
                    return &layout;
            }
            return nullptr;
        }

        float blockBytesPerWeight(size_t block_bytes, size_t block_elements)
        {
            return static_cast<float>(block_bytes) / static_cast<float>(block_elements);
        }
    } // anonymous namespace

    float WeightMemoryEstimator::getNativeBytesPerWeight(const std::string &quant_type)
    {
        if (quant_type == "F16" || quant_type == "FP16" || quant_type == "BF16")
            return 2.0f;
        if (quant_type == "F32")
            return 4.0f;

        if (const auto *layout = findNativeBlockLayout(quant_type))
            return blockBytesPerWeight(layout->block_bytes, layout->block_elements);

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
        if (quant_type == "F16" || quant_type == "FP16" || quant_type == "BF16")
            return 2.0f;
        if (quant_type == "F32")
            return 4.0f;

        if (const auto *layout = findGPUNativeVNNILayout(quant_type))
        {
            const int scale_bytes = sizeof(uint16_t);
            const int min_bytes = layout->is_asymmetric ? static_cast<int>(sizeof(uint16_t)) : 0;
            const int emin_bytes = layout->has_emins ? static_cast<int>(sizeof(uint32_t)) : 0;
            return static_cast<float>(layout->payload_bytes_per_32 + scale_bytes + min_bytes + emin_bytes) / 32.0f;
        }

        if (quant_type == "Q8_K")
            return blockBytesPerWeight(sizeof(Q8_KBlock), Q8_KBlock::BLOCK_SIZE);

        return getCUDAPackedBytesPerWeight(K);
    }

    float WeightMemoryEstimator::getCPUPackedBytesPerWeight(const std::string &quant_type)
    {
        // CPU VNNI packing expands quantized weights for efficient VNNI/AVX512 processing.
        // Q8_0 → int8 with block scales, ~1.125 bytes/weight
        // Q4_0 → dequant to int8 for VNNI, ~1.125 bytes/weight (after expansion)
        // IQ4_NL → dequant to int8, ~1.125 bytes/weight
        if (findNativeBlockLayout(quant_type) != nullptr)
        {
            return 1.125f; // int8 packed with scale overhead
        }
        if (quant_type == "F16" || quant_type == "FP16" || quant_type == "BF16")
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
