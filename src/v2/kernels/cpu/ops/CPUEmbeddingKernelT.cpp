/**
 * @file CPUEmbeddingKernelT.cpp
 * @brief CPU embedding kernel implementation
 *
 * @author David Sanftenberg
 */

#include "CPUEmbeddingKernelT.h"
#include "../../../tensors/SIMDHelpers.h"
#include "../../../tensors/FP16Utils.h"
#include "../../../utils/KernelProfiler.h"
#include <cstring>
#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace llaminar2
{

    namespace
    {
        struct EmbedRepackCacheKey
        {
            const void *raw_ptr = nullptr;
            size_t rows = 0;
            int d_model = 0;
            TensorType type = TensorType::FP32;

            bool operator==(const EmbedRepackCacheKey &other) const
            {
                return raw_ptr == other.raw_ptr &&
                       rows == other.rows &&
                       d_model == other.d_model &&
                       type == other.type;
            }
        };

        struct EmbedRepackCacheKeyHash
        {
            size_t operator()(const EmbedRepackCacheKey &k) const
            {
                size_t h = std::hash<const void *>{}(k.raw_ptr);
                h ^= std::hash<size_t>{}(k.rows) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= std::hash<int>{}(k.d_model) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= std::hash<int>{}(static_cast<int>(k.type)) + 0x9e3779b9 + (h << 6) + (h >> 2);
                return h;
            }
        };

        std::shared_ptr<const EmbedQ8RepackResult> getOrCreateCachedEmbeddingRepack(
            const TensorBase *embed_table,
            int d_model)
        {
            static std::mutex cache_mutex;
            static std::unordered_map<EmbedRepackCacheKey,
                                      std::shared_ptr<const EmbedQ8RepackResult>,
                                      EmbedRepackCacheKeyHash>
                cache;

            const void *raw_ptr = embed_table->raw_data();
            if (!raw_ptr)
            {
                raw_ptr = static_cast<const void *>(embed_table);
            }

            const EmbedRepackCacheKey key{
                raw_ptr,
                embed_table->rows(),
                d_model,
                embed_table->native_type()};

            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                auto it = cache.find(key);
                if (it != cache.end())
                {
                    return it->second;
                }
            }

            auto repacked = std::make_shared<EmbedQ8RepackResult>(repackEmbeddingToQ8(embed_table, d_model));

            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                cache[key] = repacked;
            }

            return repacked;
        }
    } // namespace

    template <typename TensorT>
    bool CPUEmbeddingKernelT<TensorT>::apply(
        const float *embed_data,
        const int *token_ids,
        int num_tokens,
        int d_model,
        float *output,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if constexpr (std::is_same_v<TensorT, FP32Tensor>)
        {
            (void)mpi_ctx;
            (void)device_idx;

            if (!embed_data || !token_ids || !output)
            {
                return false;
            }

            // Direct memcpy for FP32 output
            for (int i = 0; i < num_tokens; ++i)
            {
                int token_id = token_ids[i];
                std::memcpy(output + i * d_model,
                            embed_data + token_id * d_model,
                            d_model * sizeof(float));
            }

            return true;
        }
        else
        {
            return false;
        }
    }

    template <typename TensorT>
    bool CPUEmbeddingKernelT<TensorT>::apply_bf16(
        const float *embed_data,
        const int *token_ids,
        int num_tokens,
        int d_model,
        uint16_t *output,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if constexpr (std::is_same_v<TensorT, BF16Tensor>)
        {
            (void)mpi_ctx;
            (void)device_idx;

            if (!embed_data || !token_ids || !output)
            {
                return false;
            }

            // Lookup FP32 embeddings, convert to BF16
            for (int i = 0; i < num_tokens; ++i)
            {
                int token_id = token_ids[i];
                const float *embed_row = embed_data + token_id * d_model;
                uint16_t *out_row = output + i * d_model;

                for (int j = 0; j < d_model; ++j)
                {
                    out_row[j] = simd::fp32_to_bf16(embed_row[j]);
                }
            }

            return true;
        }
        else
        {
            return false;
        }
    }

    template <typename TensorT>
    bool CPUEmbeddingKernelT<TensorT>::apply_fp16(
        const float *embed_data,
        const int *token_ids,
        int num_tokens,
        int d_model,
        uint16_t *output,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if constexpr (std::is_same_v<TensorT, FP16Tensor>)
        {
            (void)mpi_ctx;
            (void)device_idx;

            if (!embed_data || !token_ids || !output)
            {
                return false;
            }

            // Lookup FP32 embeddings, convert to FP16
            for (int i = 0; i < num_tokens; ++i)
            {
                int token_id = token_ids[i];
                const float *embed_row = embed_data + token_id * d_model;
                uint16_t *out_row = output + i * d_model;

                for (int j = 0; j < d_model; ++j)
                {
                    out_row[j] = simd::fp32_to_fp16(embed_row[j]);
                }
            }

            return true;
        }
        else
        {
            return false;
        }
    }

    template <typename TensorT>
    bool CPUEmbeddingKernelT<TensorT>::apply_q8_1(
        const float *embed_data,
        const int *token_ids,
        int num_tokens,
        int d_model,
        void *output,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if constexpr (std::is_same_v<TensorT, Q8_1Tensor>)
        {
            (void)mpi_ctx;
            (void)device_idx;

            if (!embed_data || !token_ids || !output)
            {
                return false;
            }

            // Q8_1 block count per row: d_model / 32 (32 elements per block)
            const int blocks_per_row = d_model / 32;
            if (d_model % 32 != 0)
            {
                return false; // d_model must be multiple of 32 for Q8_1
            }

            Q8_1Block *output_blocks = static_cast<Q8_1Block *>(output);

            // Lookup FP32 embeddings, quantize to Q8_1
            for (int i = 0; i < num_tokens; ++i)
            {
                int token_id = token_ids[i];
                const float *embed_row = embed_data + token_id * d_model;
                Q8_1Block *out_row = output_blocks + i * blocks_per_row;

                // Quantize FP32 row to Q8_1 blocks using SIMD primitives
                simd::quantize_fp32_to_q8_1_blocks(embed_row, out_row, d_model);
            }

            return true;
        }
        else
        {
            return false;
        }
    }

    template <typename TensorT>
    bool CPUEmbeddingKernelT<TensorT>::apply_tensor(
        const TensorBase *embed_table,
        const int *token_ids,
        int num_tokens,
        int d_model,
        TensorBase *output,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        KERNEL_PROFILE_SCOPE(KernelType::EMBEDDING);
        if (!embed_table || !token_ids || !output)
        {
            return false;
        }

        // =====================================================================
        // Fast path: FP32 embedding table — no dequantization needed
        // =====================================================================
        if (embed_table->native_type() == TensorType::FP32)
        {
            const float *embed_data = embed_table->data();
            if (!embed_data)
                return false;

            if constexpr (std::is_same_v<TensorT, FP32Tensor>)
            {
                if (output->native_type() != TensorType::FP32)
                    return false;
                return apply(embed_data, token_ids, num_tokens, d_model,
                             output->mutable_data(), mpi_ctx, device_idx);
            }
            else if constexpr (std::is_same_v<TensorT, BF16Tensor>)
            {
                if (output->native_type() != TensorType::BF16)
                    return false;
                auto *bf16_output = dynamic_cast<BF16Tensor *>(output);
                if (!bf16_output)
                    return false;
                return apply_bf16(embed_data, token_ids, num_tokens, d_model,
                                  bf16_output->mutable_typed_data(), mpi_ctx, device_idx);
            }
            else if constexpr (std::is_same_v<TensorT, FP16Tensor>)
            {
                if (output->native_type() != TensorType::FP16)
                    return false;
                auto *fp16_output = dynamic_cast<FP16Tensor *>(output);
                if (!fp16_output)
                    return false;
                return apply_fp16(embed_data, token_ids, num_tokens, d_model,
                                  fp16_output->mutable_typed_data(), mpi_ctx, device_idx);
            }
            else if constexpr (std::is_same_v<TensorT, Q8_1Tensor>)
            {
                if (output->native_type() != TensorType::Q8_1)
                    return false;
                auto *q8_output = dynamic_cast<Q8_1Tensor *>(output);
                if (!q8_output)
                    return false;
                return apply_q8_1(embed_data, token_ids, num_tokens, d_model,
                                  q8_output->mutable_typed_data(), mpi_ctx, device_idx);
            }
            else
            {
                return false;
            }
        }

        // =====================================================================
        // Quantized path: repack to EmbedQ8 via IINT8Unpackable, dequant per-row
        // =====================================================================
        const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(embed_table);
        if (!unpackable)
        {
            LOG_ERROR("[CPUEmbeddingKernelT] Embedding table type "
                      << tensorTypeName(embed_table->native_type())
                      << " is not FP32 and does not implement IINT8Unpackable");
            return false;
        }

        // Cache the repacked EmbedQ8 data (one-time cost per embedding table)
        if (cached_embed_table_ != embed_table || !cached_repack_)
        {
            cached_repack_ = getOrCreateCachedEmbeddingRepack(embed_table, d_model);
            cached_embed_table_ = embed_table;
        }

        const EmbedQ8Block *blocks = reinterpret_cast<const EmbedQ8Block *>(cached_repack_->data.data());
        const size_t blocks_per_row = cached_repack_->blocks_per_row;

        if constexpr (std::is_same_v<TensorT, FP32Tensor>)
        {
            if (output->native_type() != TensorType::FP32)
                return false;
            float *out = output->mutable_data();

            for (int i = 0; i < num_tokens; ++i)
            {
                int token_id = token_ids[i];
                float *out_row = out + static_cast<size_t>(i) * d_model;
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    const EmbedQ8Block &blk = blocks[token_id * blocks_per_row + b];
                    float scale = fp16_to_fp32(blk.d);
                    float min_val = fp16_to_fp32(blk.m);
                    int base = static_cast<int>(b) * 32;
                    int count = std::min(32, d_model - base);
                    for (int j = 0; j < count; ++j)
                        out_row[base + j] = static_cast<float>(blk.qs[j]) * scale + min_val;
                }
            }
            return true;
        }
        else if constexpr (std::is_same_v<TensorT, BF16Tensor>)
        {
            if (output->native_type() != TensorType::BF16)
                return false;
            auto *bf16_output = dynamic_cast<BF16Tensor *>(output);
            if (!bf16_output)
                return false;
            uint16_t *out = bf16_output->mutable_typed_data();

            for (int i = 0; i < num_tokens; ++i)
            {
                int token_id = token_ids[i];
                uint16_t *out_row = out + static_cast<size_t>(i) * d_model;
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    const EmbedQ8Block &blk = blocks[token_id * blocks_per_row + b];
                    float scale = fp16_to_fp32(blk.d);
                    float min_val = fp16_to_fp32(blk.m);
                    int base = static_cast<int>(b) * 32;
                    int count = std::min(32, d_model - base);
                    for (int j = 0; j < count; ++j)
                    {
                        float val = static_cast<float>(blk.qs[j]) * scale + min_val;
                        out_row[base + j] = simd::fp32_to_bf16(val);
                    }
                }
            }
            return true;
        }
        else if constexpr (std::is_same_v<TensorT, FP16Tensor>)
        {
            if (output->native_type() != TensorType::FP16)
                return false;
            auto *fp16_output = dynamic_cast<FP16Tensor *>(output);
            if (!fp16_output)
                return false;
            uint16_t *out = fp16_output->mutable_typed_data();

            for (int i = 0; i < num_tokens; ++i)
            {
                int token_id = token_ids[i];
                uint16_t *out_row = out + static_cast<size_t>(i) * d_model;
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    const EmbedQ8Block &blk = blocks[token_id * blocks_per_row + b];
                    float scale = fp16_to_fp32(blk.d);
                    float min_val = fp16_to_fp32(blk.m);
                    int base = static_cast<int>(b) * 32;
                    int count = std::min(32, d_model - base);
                    for (int j = 0; j < count; ++j)
                    {
                        float val = static_cast<float>(blk.qs[j]) * scale + min_val;
                        out_row[base + j] = simd::fp32_to_fp16(val);
                    }
                }
            }
            return true;
        }
        else if constexpr (std::is_same_v<TensorT, Q8_1Tensor>)
        {
            if (output->native_type() != TensorType::Q8_1)
                return false;
            auto *q8_output = dynamic_cast<Q8_1Tensor *>(output);
            if (!q8_output)
                return false;

            // Dequant each token row to FP32 temp, then quantize to Q8_1
            const int q8_blocks_per_row = d_model / 32;
            if (d_model % 32 != 0)
                return false;

            Q8_1Block *output_blocks = q8_output->mutable_typed_data();
            alignas(64) float row_buf[8192]; // d_model ≤ 8192 for all current models

            for (int i = 0; i < num_tokens; ++i)
            {
                int token_id = token_ids[i];
                // Dequant from EmbedQ8 to FP32 temp
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    const EmbedQ8Block &blk = blocks[token_id * blocks_per_row + b];
                    float scale = fp16_to_fp32(blk.d);
                    float min_val = fp16_to_fp32(blk.m);
                    int base = static_cast<int>(b) * 32;
                    int count = std::min(32, d_model - base);
                    for (int j = 0; j < count; ++j)
                        row_buf[base + j] = static_cast<float>(blk.qs[j]) * scale + min_val;
                }
                // Quantize FP32 row → Q8_1 blocks
                simd::quantize_fp32_to_q8_1_blocks(
                    row_buf, output_blocks + i * q8_blocks_per_row, d_model);
            }
            return true;
        }
        else
        {
            return false;
        }
    }

    // Explicit instantiations
    template class CPUEmbeddingKernelT<FP32Tensor>;
    template class CPUEmbeddingKernelT<BF16Tensor>;
    template class CPUEmbeddingKernelT<FP16Tensor>;
    template class CPUEmbeddingKernelT<Q8_1Tensor>;

} // namespace llaminar2
