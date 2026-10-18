/**
 * @file CPUEmbeddingKernelT.cpp
 * @brief CPU native-source embedding lookup with explicit activation conversion
 *
 * Quantized source bytes and original scales remain immutable. Only requested
 * vocabulary rows are decoded; output precision belongs to the activation tensor.
 * No source-table requantization or global derived-weight cache is retained.
 *
 * @author David Sanftenberg
 */

#include "CPUEmbeddingKernelT.h"
#include "../../../tensors/SIMDHelpers.h"
#include "../../../tensors/TensorSlice.h"
#include "../../../tensors/FP16Utils.h"
#include "../../../utils/KernelProfiler.h"
#include "../../../utils/PerfStatsCollector.h"
#include <cstring>
#include <string>
#include <type_traits>

namespace llaminar2
{

    namespace
    {
        /**
         * @brief Publish proof that a compact verifier lookup used one grouped API call.
         *
         * Embedding rows are independent, so the economical grouped CPU path is a
         * single loop over M token IDs. M=1 calls are the serial oracle and do not
         * emit this counter.
         */
        void recordCPUGroupedEmbeddingCall(
            const TensorBase *embed_table,
            const TensorBase *output,
            int num_tokens,
            int d_model,
            const char *weight_route)
        {
            if (num_tokens < 2)
                return;

            PerfStatsCollector::addCounter(
                "kernel",
                "cpu_embedding_grouped_verifier_rows_calls",
                1.0,
                "verifier",
                DeviceId::cpu().to_string(),
                {{"weight_format", tensorTypeName(embed_table->native_type())},
                 {"output_format", tensorTypeName(output->native_type())},
                 {"verifier_rows", std::to_string(num_tokens)},
                 {"d_model", std::to_string(d_model)},
                 {"weight_route", weight_route},
                 {"token_source", "host"},
                 {"invocation_policy", "single_grouped_call"}});
        }
    } // namespace

    template <typename TensorT>
    bool CPUEmbeddingKernelT<TensorT>::apply(
        const float *embed_data,
        const int *token_ids,
        int num_tokens,
        int d_model,
        float *output,
        const IMPIContext *mpi_ctx,
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

            // Direct memcpy for FP32 output (with vocab-parallel sharding support)
            if (!vocab_parallel_)
            {
                for (int i = 0; i < num_tokens; ++i)
                {
                    int token_id = token_ids[i];
                    std::memcpy(output + i * d_model,
                                embed_data + token_id * d_model,
                                d_model * sizeof(float));
                }
                return true;
            }

            for (int i = 0; i < num_tokens; ++i)
            {
                int token_id = token_ids[i];
                int local_id = token_id - vocab_offset_;
                if (local_id < 0 || local_id >= local_vocab_size_)
                {
                    // Token belongs to a different rank's shard — output zeros
                    std::memset(output + i * d_model, 0, d_model * sizeof(float));
                    continue;
                }
                std::memcpy(output + i * d_model,
                            embed_data + local_id * d_model,
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
        const IMPIContext *mpi_ctx,
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

            // Lookup FP32 embeddings, convert to BF16 (with vocab-parallel sharding support)
            for (int i = 0; i < num_tokens; ++i)
            {
                int token_id = token_ids[i];
                int local_id = vocab_parallel_ ? token_id - vocab_offset_ : token_id;
                if (vocab_parallel_ && (local_id < 0 || local_id >= local_vocab_size_))
                {
                    // Token belongs to a different rank's shard — output zeros
                    std::memset(output + i * d_model, 0, d_model * sizeof(uint16_t));
                    continue;
                }
                const float *embed_row = embed_data + local_id * d_model;
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
        const IMPIContext *mpi_ctx,
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

            // Lookup FP32 embeddings, convert to FP16 (with vocab-parallel sharding support)
            for (int i = 0; i < num_tokens; ++i)
            {
                int token_id = token_ids[i];
                int local_id = vocab_parallel_ ? token_id - vocab_offset_ : token_id;
                if (vocab_parallel_ && (local_id < 0 || local_id >= local_vocab_size_))
                {
                    // Token belongs to a different rank's shard — output zeros
                    std::memset(output + i * d_model, 0, d_model * sizeof(uint16_t));
                    continue;
                }
                const float *embed_row = embed_data + local_id * d_model;
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
        const IMPIContext *mpi_ctx,
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

            // Lookup FP32 embeddings, quantize to Q8_1 (with vocab-parallel sharding support)
            for (int i = 0; i < num_tokens; ++i)
            {
                int token_id = token_ids[i];
                int local_id = vocab_parallel_ ? token_id - vocab_offset_ : token_id;
                Q8_1Block *out_row = output_blocks + i * blocks_per_row;

                if (vocab_parallel_ && (local_id < 0 || local_id >= local_vocab_size_))
                {
                    // Token belongs to a different rank's shard — output zero blocks
                    std::memset(out_row, 0, blocks_per_row * sizeof(Q8_1Block));
                    continue;
                }

                const float *embed_row = embed_data + local_id * d_model;

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
        const IMPIContext *mpi_ctx,
        int device_idx)
    {
        KERNEL_PROFILE_SCOPE(KernelType::EMBEDDING);
        if (!embed_table || !token_ids || !output || num_tokens <= 0 || d_model <= 0)
        {
            return false;
        }

        if (embed_table->cols() != static_cast<size_t>(d_model) ||
            output->numel() < static_cast<size_t>(num_tokens) * static_cast<size_t>(d_model))
            return false;

        // Compute vocab-parallel sharding state
        // When the embedding table is column-parallel sharded, each rank holds
        // vocab_size/tp_degree rows. Tokens outside the local range must produce zeros;
        // the subsequent AllReduce sums partial results across ranks.
        if (explicit_vocab_range_)
        {
            if (local_vocab_size_ <= 0)
                local_vocab_size_ = static_cast<int>(embed_table->rows());
        }
        else
        {
            local_vocab_size_ = static_cast<int>(embed_table->rows());
            vocab_offset_ = 0;
            if (mpi_ctx && mpi_ctx->world_size() > 1)
            {
                vocab_offset_ = mpi_ctx->rank() * local_vocab_size_;
            }
        }
        vocab_parallel_ = explicit_vocab_range_ ||
                          (mpi_ctx && mpi_ctx->world_size() > 1) ||
                          vocab_offset_ != 0 ||
                          local_vocab_size_ < static_cast<int>(embed_table->rows());

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
                const bool ok = apply(embed_data, token_ids, num_tokens, d_model,
                                      output->mutable_data(), mpi_ctx, device_idx);
                if (ok)
                    recordCPUGroupedEmbeddingCall(embed_table, output, num_tokens, d_model, "direct_fp32");
                return ok;
            }
            else if constexpr (std::is_same_v<TensorT, BF16Tensor>)
            {
                if (output->native_type() != TensorType::BF16)
                    return false;
                auto *bf16_output = dynamic_cast<BF16Tensor *>(output);
                if (!bf16_output)
                    return false;
                const bool ok = apply_bf16(embed_data, token_ids, num_tokens, d_model,
                                           bf16_output->mutable_typed_data(), mpi_ctx, device_idx);
                if (ok)
                    recordCPUGroupedEmbeddingCall(embed_table, output, num_tokens, d_model, "direct_fp32");
                return ok;
            }
            else if constexpr (std::is_same_v<TensorT, FP16Tensor>)
            {
                if (output->native_type() != TensorType::FP16)
                    return false;
                auto *fp16_output = dynamic_cast<FP16Tensor *>(output);
                if (!fp16_output)
                    return false;
                const bool ok = apply_fp16(embed_data, token_ids, num_tokens, d_model,
                                           fp16_output->mutable_typed_data(), mpi_ctx, device_idx);
                if (ok)
                    recordCPUGroupedEmbeddingCall(embed_table, output, num_tokens, d_model, "direct_fp32");
                return ok;
            }
            else if constexpr (std::is_same_v<TensorT, Q8_1Tensor>)
            {
                if (output->native_type() != TensorType::Q8_1)
                    return false;
                auto *q8_output = dynamic_cast<Q8_1Tensor *>(output);
                if (!q8_output)
                    return false;
                const bool ok = apply_q8_1(embed_data, token_ids, num_tokens, d_model,
                                           q8_output->mutable_typed_data(), mpi_ctx, device_idx);
                if (ok)
                    recordCPUGroupedEmbeddingCall(embed_table, output, num_tokens, d_model, "direct_fp32");
                return ok;
            }
            else
            {
                return false;
            }
        }

        // =====================================================================
        // Native floating path: preserve FP16/BF16 table precision exactly
        // =====================================================================
        if (embed_table->native_type() == TensorType::FP16 ||
            embed_table->native_type() == TensorType::BF16)
        {
            constexpr int kMaxStackEmbeddingWidth = 8192;
            if (d_model > kMaxStackEmbeddingWidth)
            {
                LOG_ERROR("[CPUEmbeddingKernelT] Floating embedding width " << d_model
                                                                            << " exceeds supported stack row width "
                                                                            << kMaxStackEmbeddingWidth);
                return false;
            }

            if constexpr (std::is_same_v<TensorT, FP32Tensor>)
            {
                if (output->native_type() != TensorType::FP32)
                    return false;
            }
            else if constexpr (std::is_same_v<TensorT, BF16Tensor>)
            {
                if (output->native_type() != TensorType::BF16)
                    return false;
            }
            else if constexpr (std::is_same_v<TensorT, FP16Tensor>)
            {
                if (output->native_type() != TensorType::FP16)
                    return false;
            }
            else if constexpr (std::is_same_v<TensorT, Q8_1Tensor>)
            {
                if (output->native_type() != TensorType::Q8_1 || d_model % 32 != 0)
                    return false;
            }
            else
            {
                return false;
            }

            alignas(64) float row_buffer[kMaxStackEmbeddingWidth];
            auto *output_bytes = static_cast<uint8_t *>(output->raw_mutable_data());
            const size_t row_bytes = output->size_bytes() / static_cast<size_t>(num_tokens);

            for (int row = 0; row < num_tokens; ++row)
            {
                const int local_id = token_ids[row] - vocab_offset_;
                auto *destination = output_bytes + static_cast<size_t>(row) * row_bytes;
                if (local_id < 0 || local_id >= local_vocab_size_)
                {
                    std::memset(destination, 0, row_bytes);
                    continue;
                }

                embed_table->to_fp32_row(static_cast<size_t>(local_id), row_buffer);
                if constexpr (std::is_same_v<TensorT, FP32Tensor>)
                {
                    std::memcpy(destination, row_buffer, static_cast<size_t>(d_model) * sizeof(float));
                }
                else if constexpr (std::is_same_v<TensorT, BF16Tensor>)
                {
                    auto *typed_destination = reinterpret_cast<uint16_t *>(destination);
                    for (int column = 0; column < d_model; ++column)
                        typed_destination[column] = simd::fp32_to_bf16(row_buffer[column]);
                }
                else if constexpr (std::is_same_v<TensorT, FP16Tensor>)
                {
                    auto *typed_destination = reinterpret_cast<uint16_t *>(destination);
                    for (int column = 0; column < d_model; ++column)
                        typed_destination[column] = simd::fp32_to_fp16(row_buffer[column]);
                }
                else if constexpr (std::is_same_v<TensorT, Q8_1Tensor>)
                {
                    simd::quantize_fp32_to_q8_1_blocks(
                        row_buffer,
                        reinterpret_cast<Q8_1Block *>(destination),
                        d_model);
                }
            }

            recordCPUGroupedEmbeddingCall(
                embed_table,
                output,
                num_tokens,
                d_model,
                embed_table->native_type() == TensorType::FP16 ? "native_fp16" : "native_bf16");
            return true;
        }

        // =====================================================================
        // Quantized sources keep the exact loader-owned native row. Looking
        // up a row is decoding, not another quantization/preparation operation.
        if (!NativeEmbeddingLayout::supports(embed_table->native_type()) ||
            !embed_table->raw_data() || embed_table->cols() != static_cast<size_t>(d_model))
        {
            LOG_ERROR("[CPUEmbeddingKernelT] Native embedding source is unavailable or incompatible");
            return false;
        }
        if constexpr (std::is_same_v<TensorT, FP32Tensor>)
        {
            if (output->native_type() != TensorType::FP32)
                return false;
            float *out = output->mutable_data();
            for (int row = 0; row < num_tokens; ++row)
            {
                const int64_t local_id = static_cast<int64_t>(token_ids[row]) - vocab_offset_;
                float *destination = out + static_cast<size_t>(row) * d_model;
                if (local_id < 0 || local_id >= local_vocab_size_)
                    std::memset(destination, 0, static_cast<size_t>(d_model) * sizeof(float));
                else
                    embed_table->to_fp32_row(static_cast<size_t>(local_id), destination);
            }
        }
        else
        {
            // Slice wrappers retain an already materialized native view. The
            // block decoder uses that view's own row offsets, never global TP
            // coordinates or the original parent's unsliced row zero.
            const TensorBase *native_source = embed_table;
            while (const auto *slice = dynamic_cast<const TensorSlice *>(native_source))
                native_source = slice->inner();
            const auto *decoder = dynamic_cast<const ITensorGemmTileDataProvider *>(native_source);
            const auto native_layout = NativeEmbeddingLayout::forSource(
                embed_table->native_type(), embed_table->rows(), d_model);
            // The decoder's logical tile may be a K32 sub-block of the
            // physical K256 source block. Its own tile size owns iteration;
            // native_layout owns source storage geometry, not decode width.
            if (!decoder || decoder->block_size() == 0u ||
                decoder->block_size() > 256u || decoder->block_size() % 32u != 0u ||
                native_layout.block_elements % decoder->block_size() != 0u ||
                decoder->decoder_cols() != static_cast<size_t>(d_model))
                return false;
            if constexpr (std::is_same_v<TensorT, BF16Tensor>)
            {
                if (output->native_type() != TensorType::BF16) return false;
            }
            else if constexpr (std::is_same_v<TensorT, FP16Tensor>)
            {
                if (output->native_type() != TensorType::FP16) return false;
            }
            else if constexpr (std::is_same_v<TensorT, Q8_1Tensor>)
            {
                if (output->native_type() != TensorType::Q8_1 || d_model % 32 != 0) return false;
            }
            else return false;
            auto *typed_output = dynamic_cast<TensorT *>(output);
            if (!typed_output) return false;
            alignas(64) float source_block[256];
            for (int row = 0; row < num_tokens; ++row)
            {
                const int64_t local_id = static_cast<int64_t>(token_ids[row]) - vocab_offset_;
                if (local_id < 0 || local_id >= local_vocab_size_)
                {
                    const size_t bytes_per_row = std::is_same_v<TensorT, Q8_1Tensor>
                        ? static_cast<size_t>(d_model / 32) * sizeof(Q8_1Block)
                        : static_cast<size_t>(d_model) * sizeof(uint16_t);
                    std::memset(static_cast<uint8_t *>(output->raw_mutable_data()) +
                                    static_cast<size_t>(row) * bytes_per_row, 0, bytes_per_row);
                    continue;
                }
                const size_t block_values = decoder->block_size();
                for (size_t block = 0, first = 0; first < static_cast<size_t>(d_model);
                     ++block, first += block_values)
                {
                    const size_t count = std::min(block_values, static_cast<size_t>(d_model) - first);
                    decoder->decode_block_at(static_cast<size_t>(local_id), block, source_block);
                    if constexpr (std::is_same_v<TensorT, BF16Tensor>)
                    {
                        auto *destination = typed_output->mutable_typed_data() +
                                            static_cast<size_t>(row) * d_model + first;
                        for (size_t col = 0; col < count; ++col)
                            destination[col] = simd::fp32_to_bf16(source_block[col]);
                    }
                    else if constexpr (std::is_same_v<TensorT, FP16Tensor>)
                    {
                        auto *destination = typed_output->mutable_typed_data() +
                                            static_cast<size_t>(row) * d_model + first;
                        for (size_t col = 0; col < count; ++col)
                            destination[col] = simd::fp32_to_fp16(source_block[col]);
                    }
                    else if constexpr (std::is_same_v<TensorT, Q8_1Tensor>)
                    {
                        auto *destination = typed_output->mutable_typed_data() +
                                            static_cast<size_t>(row) * (d_model / 32) + first / 32;
                        simd::quantize_fp32_to_q8_1_blocks(source_block, destination, static_cast<int>(count));
                    }
                }
            }
        }
        recordCPUGroupedEmbeddingCall(embed_table, output, num_tokens, d_model, "native_quantized");
        return true;
    }

    // Explicit instantiations
    template class CPUEmbeddingKernelT<FP32Tensor>;
    template class CPUEmbeddingKernelT<BF16Tensor>;
    template class CPUEmbeddingKernelT<FP16Tensor>;
    template class CPUEmbeddingKernelT<Q8_1Tensor>;

} // namespace llaminar2
