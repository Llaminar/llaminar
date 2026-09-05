#include "planning/WeightMemoryEstimator.h"
#include "planning/ModelMemoryProfile.h"
#include "config/GDNHeadAssignment.h"
#include "execution/local_execution/graph/SchemaFactoryRegistry.h"
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

        /** @brief Scale an integral tensor measure by an exact expert fraction. */
        size_t selectedExpertFraction(
            size_t complete,
            int selected,
            int total)
        {
            if (selected == 0 || complete == 0)
                return 0;
            if (selected == total)
                return complete;

            const size_t selected_size = static_cast<size_t>(selected);
            const size_t total_size = static_cast<size_t>(total);
            const size_t complete_quotient = complete / total_size;
            const size_t complete_remainder = complete % total_size;
            if (complete_quotient >
                std::numeric_limits<size_t>::max() / selected_size)
                throw std::overflow_error(
                    "Routed-expert memory fraction overflows size_t");
            const size_t quotient_bytes =
                complete_quotient * selected_size;
            if (complete_remainder >
                (std::numeric_limits<size_t>::max() - total_size + 1u) /
                    selected_size)
            {
                throw std::overflow_error(
                    "Routed-expert memory remainder overflows size_t");
            }
            const size_t remainder_bytes =
                (complete_remainder * selected_size + total_size - 1u) /
                total_size;
            if (quotient_bytes >
                std::numeric_limits<size_t>::max() - remainder_bytes)
            {
                throw std::overflow_error(
                    "Routed-expert memory total overflows size_t");
            }

            /*
             * Expert parents are expert-major GGUF tensors, so their byte and
             * element counts are normally exactly divisible. Round upward for
             * malformed or future padded layouts: admission may be
             * conservative, but it must never undercount a resident slice.
             */
            return quotient_bytes + remainder_bytes;
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

        /** @brief Return one equal TP shard's exact row count. */
        size_t equalShardRows(
            size_t rows,
            int shard_index,
            int total_shards)
        {
            if (total_shards <= 1)
                return rows;
            if (shard_index < 0 || shard_index >= total_shards)
            {
                throw std::invalid_argument(
                    "Weight memory estimator received an invalid TP shard index");
            }
            const size_t degree = static_cast<size_t>(total_shards);
            const size_t index = static_cast<size_t>(shard_index);
            return rows / degree + (index < rows % degree ? 1u : 0u);
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

        /** @brief One typed interval in a model-owned logical dimension. */
        struct LogicalShardInterval
        {
            size_t start = 0u;
            size_t count = 0u;
            size_t total = 0u;
        };

        /** @brief Divide one semantic dimension while keeping remainders whole. */
        LogicalShardInterval equalLogicalShardInterval(
            size_t total,
            int shard_index,
            int total_shards,
            const std::string &tensor_name)
        {
            if (total == 0u || total_shards <= 0 || shard_index < 0 ||
                shard_index >= total_shards)
            {
                throw std::invalid_argument(
                    "Weight memory estimator cannot divide the semantic dimension for " +
                    tensor_name);
            }
            const size_t degree = static_cast<size_t>(total_shards);
            const size_t index = static_cast<size_t>(shard_index);
            const size_t quotient = total / degree;
            const size_t remainder = total % degree;
            const size_t count = quotient + (index < remainder ? 1u : 0u);
            if (count == 0u)
            {
                throw std::invalid_argument(
                    "Tensor-parallel degree exceeds the semantic dimension for " +
                    tensor_name);
            }
            return {
                .start = index * quotient + std::min(index, remainder),
                .count = count,
                .total = total,
            };
        }

        /**
         * @brief Resolve the assignment interval named by the schema.
         *
         * This function translates only typed dimension coordinates. Tensor
         * names and sharding policy remain owned by WeightShardingConfig.
         */
        LogicalShardInterval logicalShardInterval(
            WeightDimensionType dimension,
            const ModelMemoryProfile &profile,
            const DeviceShardingAssignment &assignment,
            const std::string &tensor_name)
        {
            const auto checked = [&tensor_name](
                                     int start,
                                     int count,
                                     int total,
                                     const char *dimension_name)
            {
                if (total <= 0 || start < 0 || count <= 0 ||
                    start > total - count)
                {
                    throw std::invalid_argument(
                        "Weight memory estimator received an invalid " +
                        std::string(dimension_name) + " assignment for " +
                        tensor_name);
                }
                return LogicalShardInterval{
                    .start = static_cast<size_t>(start),
                    .count = static_cast<size_t>(count),
                    .total = static_cast<size_t>(total),
                };
            };

            switch (dimension)
            {
            case WeightDimensionType::Heads:
            case WeightDimensionType::ProportionalHeads:
            case WeightDimensionType::FusedQKVHeads:
                return checked(
                    assignment.head_start,
                    assignment.head_count,
                    profile.n_heads,
                    "query-head");
            case WeightDimensionType::KVHeads:
                return checked(
                    assignment.kv_head_start,
                    assignment.kv_head_count,
                    profile.n_kv_heads,
                    "KV-head");
            case WeightDimensionType::FFNHidden:
                return checked(
                    assignment.d_ff_start,
                    assignment.d_ff_count,
                    profile.d_ff,
                    "FFN");
            case WeightDimensionType::Vocab:
                return checked(
                    assignment.vocab_start,
                    assignment.vocab_count,
                    profile.vocab_size,
                    "vocabulary");
            case WeightDimensionType::Bias1D:
            case WeightDimensionType::None:
                break;
            }
            throw std::invalid_argument(
                "Sharded weight has no typed slice dimension in the model schema: " +
                tensor_name);
        }

        /** @brief Resolve an equal cross-rank interval in a schema-owned dimension. */
        LogicalShardInterval equalSchemaInterval(
            WeightDimensionType dimension,
            const ModelMemoryProfile &profile,
            int shard_index,
            int total_shards,
            const std::string &tensor_name)
        {
            size_t total = 0u;
            switch (dimension)
            {
            case WeightDimensionType::Heads:
            case WeightDimensionType::ProportionalHeads:
            case WeightDimensionType::FusedQKVHeads:
                total = static_cast<size_t>(profile.n_heads);
                break;
            case WeightDimensionType::KVHeads:
                total = static_cast<size_t>(profile.n_kv_heads);
                break;
            case WeightDimensionType::FFNHidden:
                total = static_cast<size_t>(profile.d_ff);
                break;
            case WeightDimensionType::Vocab:
                total = static_cast<size_t>(profile.vocab_size);
                break;
            case WeightDimensionType::Bias1D:
            case WeightDimensionType::None:
                throw std::invalid_argument(
                    "Sharded weight has no typed slice dimension in the model schema: " +
                    tensor_name);
            }
            return equalLogicalShardInterval(
                total, shard_index, total_shards, tensor_name);
        }

        /** @brief Apply a logical interval to a flat element count exactly as a slice boundary. */
        size_t intervalElementCount(
            size_t complete_elements,
            LogicalShardInterval interval)
        {
            if (interval.total == 0u ||
                interval.start > interval.total - interval.count)
            {
                throw std::invalid_argument(
                    "Weight shard interval is invalid");
            }
            const size_t begin =
                complete_elements * interval.start / interval.total;
            const size_t end =
                complete_elements * (interval.start + interval.count) /
                interval.total;
            return end - begin;
        }

        /**
         * @brief Resolve a fused-QKV participant's exact element count.
         *
         * GDN uses modulo-linked key/value ownership while full attention uses
         * independent query and KV ranges. Both are already encoded in the
         * same typed model geometry and assignment consumed by weight loading.
         */
        size_t fusedQKVElementCount(
            const TensorSizeInfo &tensor,
            const ModelMemoryProfile &profile,
            const DeviceShardingAssignment &assignment)
        {
            if (tensor.K == 0u || tensor.elements % tensor.K != 0u)
            {
                throw std::invalid_argument(
                    "Fused-QKV tensor lacks an integral matrix shape: " +
                    tensor.name);
            }
            const size_t rows = tensor.elements / tensor.K;
            if (profile.gdn_group_count > 0 &&
                profile.gdn_time_step_rank > 0 &&
                profile.gdn_state_size > 0)
            {
                const size_t expected_gdn_rows =
                    static_cast<size_t>(
                        2 * profile.gdn_group_count +
                        profile.gdn_time_step_rank) *
                    static_cast<size_t>(profile.gdn_state_size);
                if (rows == expected_gdn_rows)
                {
                    const auto gdn = GDNHeadAssignment::fromPartition(
                        profile.gdn_group_count,
                        profile.gdn_time_step_rank,
                        assignment.head_start,
                        assignment.head_count,
                        profile.n_heads);
                    return gdn.localFusedRows(profile.gdn_state_size) *
                           tensor.K;
                }
            }

            if (profile.n_heads > 0 && profile.n_kv_heads > 0 &&
                profile.head_dim > 0)
            {
                const size_t expected_attention_rows =
                    static_cast<size_t>(profile.n_heads) *
                        static_cast<size_t>(profile.head_dim) +
                    2u * static_cast<size_t>(profile.n_kv_heads) *
                        static_cast<size_t>(profile.head_dim);
                if (rows == expected_attention_rows)
                {
                    const size_t local_rows =
                        static_cast<size_t>(assignment.head_count) *
                            static_cast<size_t>(profile.head_dim) +
                        2u * static_cast<size_t>(assignment.kv_head_count) *
                            static_cast<size_t>(profile.head_dim);
                    return local_rows * tensor.K;
                }
            }

            return intervalElementCount(
                tensor.elements,
                logicalShardInterval(
                    WeightDimensionType::FusedQKVHeads,
                    profile,
                    assignment,
                    tensor.name));
        }

        /** @brief Resolve the exact participant-local logical element count. */
        size_t tensorParallelElementCount(
            const TensorSizeInfo &tensor,
            const ModelMemoryProfile &profile,
            const WeightShardingConfig &sharding,
            int shard_index,
            int total_shards,
            const std::optional<DeviceShardingAssignment> &assignment)
        {
            // Metadata-only planner fixtures can carry zero-byte layer markers
            // to identify hybrid layer kinds. They have no physical weight
            // allocation and therefore no shard geometry to resolve.
            if (tensor.elements == 0u)
                return 0u;
            const auto [mode, dimension] =
                sharding.getModeAndDimension(tensor.name);
            if (mode == WeightShardingMode::Replicate)
                return tensor.elements;

            if (mode == WeightShardingMode::ExpertIdApportioned)
            {
                if (profile.expert_count <= 0)
                {
                    throw std::invalid_argument(
                        "Expert-id sharded weight requires model expert_count metadata: " +
                        tensor.name);
                }
                const size_t local_experts = equalShardRows(
                    static_cast<size_t>(profile.expert_count),
                    shard_index,
                    total_shards);
                return selectedExpertFraction(
                    tensor.elements,
                    static_cast<int>(local_experts),
                    profile.expert_count);
            }

            if (!assignment.has_value())
            {
                /* Divide the schema's semantic axis, never the flat byte
                 * count. This keeps vocabulary rows, heads, and FFN blocks
                 * indivisible when a uniform cross-rank topology owns no
                 * DeviceShardingAssignment object. */
                const LogicalShardInterval interval = equalSchemaInterval(
                    dimension,
                    profile,
                    shard_index,
                    total_shards,
                    tensor.name);
                if (dimension == WeightDimensionType::FusedQKVHeads)
                {
                    const LogicalShardInterval kv_interval =
                        equalSchemaInterval(
                            WeightDimensionType::KVHeads,
                            profile,
                            shard_index,
                            total_shards,
                            tensor.name);
                    DeviceShardingAssignment uniform;
                    uniform.device = DeviceId::cpu();
                    uniform.local_rank = shard_index;
                    uniform.head_start = static_cast<int>(interval.start);
                    uniform.head_count = static_cast<int>(interval.count);
                    uniform.kv_head_start =
                        static_cast<int>(kv_interval.start);
                    uniform.kv_head_count =
                        static_cast<int>(kv_interval.count);
                    return fusedQKVElementCount(
                        tensor, profile, uniform);
                }
                return intervalElementCount(tensor.elements, interval);
            }

            if (dimension == WeightDimensionType::FusedQKVHeads)
                return fusedQKVElementCount(tensor, profile, *assignment);
            return intervalElementCount(
                tensor.elements,
                logicalShardInterval(
                    dimension, profile, *assignment, tensor.name));
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
            &tensor_parallel_assignment)
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

        WeightEstimate est;
        std::optional<WeightShardingConfig> sharding;
        if (total_shards > 1)
        {
            /* SchemaFactoryRegistry is the sole tensor-name policy authority.
             * Unsupported TP architecture is a configuration error, not a
             * reason to guess from substrings and silently mis-admit memory. */
            sharding = SchemaFactoryRegistry::getWeightShardingConfig(
                profile.architecture);
        }
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

            const bool routed_expert_tensor = isRoutedExpertTensor(t.name);
            if (!routed_expert_tensor && !residency.includesNonRoutedWeights())
                continue;

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
                    continue;
            }

            size_t selected_elements =
                routed_expert_tensor && residency.selectsRoutedExperts()
                    ? selectedExpertFraction(
                          t.elements,
                          selected_routed_experts,
                          profile.expert_count)
                    : t.elements;
            if (total_shards > 1 &&
                !(routed_expert_tensor && residency.selectsRoutedExperts()))
            {
                selected_elements = tensorParallelElementCount(
                    t,
                    profile,
                    *sharding,
                    shard_index,
                    total_shards,
                    tensor_parallel_assignment);
            }
            const size_t native = selectedElementBytes(
                t.native_bytes, selected_elements, t.elements);

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

                    if (!has_explicit_lm_head)
                    {
                        const auto *format =
                            native_vnni_formats::forQuantType(t.quant_type);
                        const size_t tied_lm_head_bytes =
                            format
                                ? exactGpuPackedMatrixBytes(
                                      local_rows,
                                      static_cast<size_t>(profile.d_model),
                                      *format)
                                : static_cast<size_t>(
                                      static_cast<float>(
                                          local_rows *
                                          static_cast<size_t>(profile.d_model)) *
                                      getGPUPackedBytesPerWeight(
                                          t.quant_type,
                                          static_cast<size_t>(
                                              profile.d_model)));
                        device_size += tied_lm_head_bytes;
                        est.lm_head_bytes += tied_lm_head_bytes;
                        est.tied_lm_head_bytes += tied_lm_head_bytes;
                    }
                }
                else
                {
                    if (const auto *format =
                            native_vnni_formats::forQuantType(t.quant_type);
                        format && t.K > 0)
                    {
                        const size_t rows = selected_elements / t.K;
                        device_size =
                            exactGpuPackedMatrixBytes(rows, t.K, *format);
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
            if (isLMHeadTensor(t.name))
                est.lm_head_bytes += device_size;
            est.device_bytes += device_size;
        }

        return est;
    }

} // namespace llaminar2
