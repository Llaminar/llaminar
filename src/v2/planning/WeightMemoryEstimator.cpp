#include "planning/WeightMemoryEstimator.h"
#include "planning/ModelMemoryProfile.h"
#include "kernels/common/EmbedQ8Block.h"
#include "tensors/BlockStructures.h"
#include "tensors/NativeVnniFormatInfo.h"

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

        const NativeBlockLayout *findNativeBlockLayout(const std::string &quant_type)
        {
            for (const auto &layout : kNativeBlockLayouts)
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

        bool isEmbeddingTensor(const std::string &name)
        {
            return name.find("token_embd") != std::string::npos ||
                   name.find("embed_tokens") != std::string::npos;
        }

        bool isLMHeadTensor(const std::string &name)
        {
            return name == "output.weight" ||
                   name.find("lm_head") != std::string::npos;
        }

        bool isQuantizedFormat(const std::string &quant_type)
        {
            return quant_type != "F32" &&
                   quant_type != "F16" &&
                   quant_type != "FP16" &&
                   quant_type != "BF16";
        }

        constexpr size_t kWeightPoolAlignment = 256;

        size_t alignUp(size_t bytes, size_t alignment)
        {
            return (bytes + alignment - 1) & ~(alignment - 1);
        }

        size_t exactGpuPackedMatrixBytes(
            size_t rows,
            size_t columns,
            const NativeVnniFormatInfo &format)
        {
            const NativeVnniPackedRegionSizes regions =
                nativeVnniPackedRegionSizes(rows, columns, format);

            /*
             * WeightVRAMPool starts every independently-addressable region at
             * a 256-byte boundary. Rounding each non-empty region gives the
             * exact aggregate pool contribution except for at most one final
             * trailing alignment unit, and is deliberately conservative for
             * preflight admission.
             */
            size_t bytes = alignUp(regions.payload_bytes, kWeightPoolAlignment);
            bytes += alignUp(regions.scales_bytes, kWeightPoolAlignment);
            if (regions.mins_bytes > 0)
                bytes += alignUp(regions.mins_bytes, kWeightPoolAlignment);
            if (regions.emins_bytes > 0)
                bytes += alignUp(regions.emins_bytes, kWeightPoolAlignment);
            return bytes;
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

        if (const auto *format =
                native_vnni_formats::forQuantType(quant_type))
        {
            const size_t metadata_bytes =
                sizeof(uint16_t) +
                (format->is_asymmetric ? sizeof(uint16_t) : 0) +
                (format->has_emins ? sizeof(uint32_t) : 0);
            return static_cast<float>(
                       static_cast<size_t>(format->payload_bytes) +
                       metadata_bytes) /
                   32.0f;
        }

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
        const bool has_explicit_lm_head = std::any_of(
            profile.tensors.begin(),
            profile.tensors.end(),
            [](const TensorSizeInfo &tensor)
            {
                return isLMHeadTensor(tensor.name);
            });

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
                if (isEmbeddingTensor(t.name) &&
                    isQuantizedFormat(t.quant_type) &&
                    profile.d_model > 0)
                {
                    /*
                     * GPU embedding lookup consumes the universal EmbedQ8
                     * representation, not the GEMM-native packed layout.
                     * A model with tied output weights also needs a separate
                     * GEMM representation because the LM head performs a
                     * matrix multiply over the same logical source tensor.
                     */
                    const size_t rows =
                        t.elements /
                        static_cast<size_t>(profile.d_model);
                    const size_t blocks_per_row =
                        (static_cast<size_t>(profile.d_model) + 31) / 32;
                    device_size =
                        rows * blocks_per_row * sizeof(EmbedQ8Block);
                    est.prepared_embedding_bytes += device_size;

                    if (!has_explicit_lm_head)
                    {
                        const auto *format =
                            native_vnni_formats::forQuantType(t.quant_type);
                        const size_t tied_lm_head_bytes =
                            format
                                ? exactGpuPackedMatrixBytes(
                                      rows,
                                      static_cast<size_t>(profile.d_model),
                                      *format)
                                : static_cast<size_t>(
                                      static_cast<float>(t.elements) *
                                      getGPUPackedBytesPerWeight(
                                          t.quant_type,
                                          static_cast<size_t>(
                                              profile.d_model)));
                        device_size += tied_lm_head_bytes;
                        est.tied_lm_head_bytes += tied_lm_head_bytes;
                    }
                }
                else
                {
                    size_t elements = t.elements;
                    if (total_shards > 1 && isShardedTensor(t.name))
                    {
                        elements =
                            elements / static_cast<size_t>(total_shards);
                    }

                    if (const auto *format =
                            native_vnni_formats::forQuantType(t.quant_type);
                        format && t.K > 0)
                    {
                        const size_t rows = elements / t.K;
                        device_size =
                            exactGpuPackedMatrixBytes(rows, t.K, *format);
                    }
                    else
                    {
                        const float bytes_per_weight =
                            getGPUPackedBytesPerWeight(t.quant_type, t.K);
                        device_size = static_cast<size_t>(
                            static_cast<float>(elements) *
                            bytes_per_weight);
                    }
                }
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
