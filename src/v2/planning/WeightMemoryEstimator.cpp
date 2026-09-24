#include "planning/WeightMemoryEstimator.h"
#include "planning/ModelMemoryProfile.h"
#include "planning/WeightShardGeometry.h"
#include "kernels/common/EmbedQ8Block.h"
#include "kernels/common/PreparedEmbeddingWeights.h"
#include "loaders/PreparedWeightRepresentationContract.h"
#include "tensors/BlockStructures.h"
#include "tensors/NativeVnniFormatInfo.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

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

        /**
         * @brief Identify the three routed-expert parent tensors.
         *
         * Shared experts deliberately use the `shexp` spelling and remain part
         * of the continuation's non-routed model authority. Matching the full
         * suffix avoids treating ordinary dense FFN tensors as expert slabs.
         */
        bool isRoutedExpertTensor(const std::string &name)
        {
            return name.ends_with(".ffn_gate_exps.weight") ||
                   name.ends_with(".ffn_up_exps.weight") ||
                   name.ends_with(".ffn_down_exps.weight");
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

        /** @brief Price one model-owned FP32 representation without overflow. */
        size_t preparedFp32Bytes(size_t elements, DeviceId device)
        {
            if (elements >
                std::numeric_limits<size_t>::max() / sizeof(float))
            {
                throw std::overflow_error(
                    "Prepared FP32 weight allocation overflows size_t");
            }
            const size_t bytes = elements * sizeof(float);
            return device.is_gpu()
                       ? alignUp(bytes, kWeightPoolAlignment)
                       : bytes;
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

        /** @brief Scale bytes by an exact logical-element subset without under-admission. */
        size_t selectedElementBytes(
            size_t complete_bytes,
            size_t selected_elements,
            size_t complete_elements)
        {
            if (selected_elements == 0u || complete_bytes == 0u)
                return 0u;
            if (selected_elements == complete_elements)
                return complete_bytes;
            if (complete_elements == 0u ||
                selected_elements > complete_elements)
            {
                throw std::invalid_argument(
                    "Weight shard element fraction is outside its source tensor");
            }

            const size_t quotient = complete_bytes / complete_elements;
            const size_t remainder = complete_bytes % complete_elements;
            if (quotient >
                std::numeric_limits<size_t>::max() / selected_elements)
            {
                throw std::overflow_error(
                    "Weight shard byte quotient overflows size_t");
            }
            const size_t quotient_bytes = quotient * selected_elements;
            if (remainder >
                (std::numeric_limits<size_t>::max() - complete_elements + 1u) /
                    selected_elements)
            {
                throw std::overflow_error(
                    "Weight shard byte remainder overflows size_t");
            }
            const size_t remainder_bytes =
                (remainder * selected_elements + complete_elements - 1u) /
                complete_elements;
            if (quotient_bytes >
                std::numeric_limits<size_t>::max() - remainder_bytes)
            {
                throw std::overflow_error(
                    "Weight shard byte total overflows size_t");
            }
            return quotient_bytes + remainder_bytes;
        }

    } // anonymous namespace

    DeviceWeightResidency::DeviceWeightResidency(
        Kind kind,
        int model_expert_count,
        std::vector<int> selected_by_layer)
        : kind_(kind),
          model_expert_count_(model_expert_count),
          selected_by_layer_(std::move(selected_by_layer))
    {
        if (kind_ == Kind::FullModel)
        {
            if (model_expert_count_ != 0 || !selected_by_layer_.empty())
                throw std::invalid_argument(
                    "Full-model weight residency cannot carry routed-expert selections");
            return;
        }
        if (model_expert_count_ <= 0 || selected_by_layer_.empty())
        {
            throw std::invalid_argument(
                "Selected routed-expert residency requires positive model geometry and layer counts");
        }
        for (size_t layer = 0; layer < selected_by_layer_.size(); ++layer)
        {
            const int count = selected_by_layer_[layer];
            if (count < 0 || count > model_expert_count_)
            {
                throw std::invalid_argument(
                    "Selected routed-expert count for layer " +
                    std::to_string(layer) + " is outside [0, " +
                    std::to_string(model_expert_count_) + "]");
            }
        }
    }

    DeviceWeightResidency
    DeviceWeightResidency::continuationWithSelectedRoutedExperts(
        int model_expert_count,
        std::vector<int> selected_by_layer)
    {
        return DeviceWeightResidency(
            Kind::ContinuationWithSelectedRoutedExperts,
            model_expert_count,
            std::move(selected_by_layer));
    }

    DeviceWeightResidency
    DeviceWeightResidency::selectedRoutedExpertsOnly(
        int model_expert_count,
        std::vector<int> selected_by_layer)
    {
        return DeviceWeightResidency(
            Kind::SelectedRoutedExpertsOnly,
            model_expert_count,
            std::move(selected_by_layer));
    }

    bool DeviceWeightResidency::includesNonRoutedWeights() const noexcept
    {
        return kind_ != Kind::SelectedRoutedExpertsOnly;
    }

    bool DeviceWeightResidency::selectsRoutedExperts() const noexcept
    {
        return kind_ != Kind::FullModel;
    }

    int DeviceWeightResidency::selectedRoutedExpertsForLayer(int layer) const
    {
        if (layer < 0 || static_cast<size_t>(layer) >= selected_by_layer_.size())
        {
            throw std::out_of_range(
                "Routed-expert residency has no owner-map entry for layer " +
                std::to_string(layer));
        }
        return selected_by_layer_[static_cast<size_t>(layer)];
    }

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

    WeightEstimate WeightMemoryEstimator::estimate(
        const ModelMemoryProfile &profile,
        DeviceId device,
        int shard_index,
        int total_shards,
        int first_layer,
        int last_layer,
        const DeviceWeightResidency &residency,
        const std::optional<DeviceShardingAssignment>
            &tensor_parallel_assignment,
        WeightComponentScope components)
    {
        if (total_shards <= 0 || shard_index < 0 ||
            shard_index >= total_shards)
        {
            throw std::invalid_argument(
                "Weight memory estimator received an invalid TP shard identity");
        }
        if (tensor_parallel_assignment.has_value() &&
            (total_shards <= 1 ||
             tensor_parallel_assignment->local_rank != shard_index ||
             tensor_parallel_assignment->device != device ||
             !tensor_parallel_assignment->isValid()))
        {
            throw std::invalid_argument(
                "Weight memory estimator TP assignment does not match its physical participant");
        }
        if (last_layer < 0)
        {
            last_layer = profile.n_layers - 1;
        }

        if (residency.selectsRoutedExperts())
        {
            if (profile.expert_count <= 0)
            {
                throw std::invalid_argument(
                    "Selected routed-expert memory planning requires model expert_count metadata");
            }
            if (profile.expert_count != residency.modelExpertCount())
            {
                throw std::invalid_argument(
                    "Routed-expert residency denominator does not match the model profile");
            }
        }

        bool owns_embedding = false;
        bool owns_terminal = false;
        switch (components)
        {
        case WeightComponentScope::EmbeddingAndTerminal:
            owns_embedding = owns_terminal = true;
            break;
        case WeightComponentScope::Embedding: owns_embedding = true; break;
        case WeightComponentScope::Terminal: owns_terminal = true; break;
        case WeightComponentScope::Intermediate:
        case WeightComponentScope::LayersOnly: break;
        default:
            throw std::invalid_argument("Weight estimate has an invalid global component scope");
        }

        WeightEstimate est;
        const WeightShardGeometryResolver geometry_resolver(
            profile, device, shard_index, total_shards, tensor_parallel_assignment);
        const bool has_explicit_lm_head = std::any_of(
            profile.tensors.begin(),
            profile.tensors.end(),
            [](const TensorSizeInfo &tensor)
            {
                return isLMHeadTensor(tensor.name);
            });

        // Each call describes one semantic runtime use. A tied vocabulary has
        // one GGUF source but distinct lookup and GEMM representations on GPU.
        // Resolving its output alias through the normal schema also preserves
        // head sharding when it differs from embedding sharding.
        const auto accumulate = [&](const TensorSizeInfo &t, bool count_native)
        {
            // Filter by PP layer range
            if (t.layer_index >= 0)
            {
                if (t.layer_index < first_layer || t.layer_index > last_layer)
                    return;
            }

            const bool routed_expert_tensor = isRoutedExpertTensor(t.name);
            if (!routed_expert_tensor && !residency.includesNonRoutedWeights())
                return;

            int selected_routed_experts = profile.expert_count;
            if (routed_expert_tensor && residency.selectsRoutedExperts())
            {
                if (t.layer_index < 0)
                {
                    throw std::invalid_argument(
                        "Routed-expert tensor lacks a model layer index: " + t.name);
                }
                selected_routed_experts =
                    residency.selectedRoutedExpertsForLayer(t.layer_index);
                if (selected_routed_experts == 0)
                    return;
            }

            const auto geometry = geometry_resolver.resolve(t,
                routed_expert_tensor && residency.selectsRoutedExperts()
                    ? std::optional<size_t>(static_cast<size_t>(selected_routed_experts))
                    : std::nullopt);
            const size_t selected_elements = geometry.elements();
            const size_t native = selectedElementBytes(
                t.native_bytes, selected_elements, t.elements);

            if (count_native)
                est.native_bytes += native;

            // Compute the exact runtime representation selected by loading.
            size_t device_size;
            const WeightRole role = inferWeightRole(t.name);
            const auto prepared_representation =
                PreparedWeightRepresentationContract::resolve(
                    role, std::string_view(t.quant_type));
            if (prepared_representation ==
                ModelPreparedWeightRepresentation::FP32)
            {
                /*
                 * These bytes replace the source codebook in the persistent
                 * execution pool. The same typed contract is consumed by
                 * WeightManager before it constructs that pool.
                 */
                device_size = preparedFp32Bytes(selected_elements, device);
            }
            else if (device.is_gpu())
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
                    if (selected_elements %
                            static_cast<size_t>(profile.d_model) !=
                        0u)
                    {
                        throw std::invalid_argument(
                            "Prepared embedding TP slice is not row integral: " +
                            t.name);
                    }
                    const size_t local_rows =
                        selected_elements /
                        static_cast<size_t>(profile.d_model);
                    device_size =
                        PreparedEmbeddingWeights::allocationBytes(
                            local_rows,
                            profile.d_model);
                    est.prepared_embedding_bytes += device_size;

                }
                else
                {
                    if (const auto *format =
                            native_vnni_formats::forQuantType(t.quant_type);
                        format && t.K > 0)
                    {
                        // Packing is a function of N and K, not merely N*K.
                        // In particular an input shard keeps every output row;
                        // reinterpreting it as fewer full-K rows can underprice
                        // row/block padding on both CUDA and ROCm.
                        const auto &matrix = geometry.matrix();
                        const size_t rows = matrix
                            ? matrix->rows * matrix->instances : selected_elements / t.K;
                        const size_t columns = matrix ? matrix->columns : t.K;
                        device_size = exactGpuPackedMatrixBytes(rows, columns, *format);
                    }
                    else
                    {
                        const float bytes_per_weight =
                            getGPUPackedBytesPerWeight(t.quant_type, t.K);
                        device_size = static_cast<size_t>(
                            static_cast<float>(selected_elements) *
                            bytes_per_weight);
                    }
                }
            }
            else
            {
                // CPU: VNNI packing
                float bytes_per_weight = getCPUPackedBytesPerWeight(t.quant_type);
                device_size = static_cast<size_t>(
                    static_cast<float>(selected_elements) * bytes_per_weight);
            }
            if (device.is_gpu() && isEmbeddingTensor(t.name) &&
                !isQuantizedFormat(t.quant_type))
            {
                // "Prepared" describes the graph-ready lookup view, not only
                // the EmbedQ8 representation. Native floating lookup reads a
                // raw device allocation prepared by WeightManager. Publish its
                // existing bytes as a component so a mirrored vocabulary view
                // is admitted beside the primary shard. This is a subset of
                // device_bytes, never a second primary allocation charge.
                est.prepared_embedding_bytes += device_size;
            }
            if (isLMHeadTensor(t.name))
                est.lm_head_bytes += device_size;
            est.device_bytes += device_size;
        };

        for (const auto &t : profile.tensors)
        {
            if (t.layer_index < 0 && components == WeightComponentScope::LayersOnly)
                continue;
            if (isEmbeddingTensor(t.name))
            {
                if (owns_embedding)
                    accumulate(t, true);
                if (owns_terminal && !has_explicit_lm_head && residency.includesNonRoutedWeights())
                {
                    auto head = t;
                    head.name = "output.weight";
                    const auto before = est.device_bytes;
                    const auto previous_head = est.lm_head_bytes;
                    accumulate(head, !owns_embedding);
                    const auto tied_bytes = est.lm_head_bytes - previous_head;
                    est.tied_lm_head_bytes += tied_bytes;
                    // CPU's existing prepared-source contract shares a tied
                    // tensor within the same participant. A GPU has a separate
                    // GEMM pool entry, including for FP32/FP16/BF16 sources.
                    if (device.is_cpu() && owns_embedding)
                        est.device_bytes = before;
                }
                continue;
            }
            if (!owns_terminal &&
                (isLMHeadTensor(t.name) || inferWeightRole(t.name) == WeightRole::OutputNorm))
                continue;
            accumulate(t, true);
        }

        return est;
    }

} // namespace llaminar2
