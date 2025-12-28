/**
 * @file CPUEmbeddingKernelT.cpp
 * @brief CPU embedding kernel implementation
 *
 * @author David Sanftenberg
 */

#include "CPUEmbeddingKernelT.h"
#include "../../../tensors/SIMDHelpers.h"
#include "../../../tensors/FP16Utils.h"
#include <cstring>
#include <type_traits>

namespace llaminar2
{

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
        if (!embed_table || !token_ids || !output)
        {
            return false;
        }

        // Get embedding table data (always FP32)
        const float *embed_data = embed_table->data();
        if (!embed_data)
        {
            return false;
        }

        if constexpr (std::is_same_v<TensorT, FP32Tensor>)
        {
            // FP32: Validate types and call typed method
            if (output->native_type() != TensorType::FP32)
            {
                return false;
            }
            return apply(embed_data, token_ids, num_tokens, d_model,
                         output->mutable_data(), mpi_ctx, device_idx);
        }
        else if constexpr (std::is_same_v<TensorT, BF16Tensor>)
        {
            // BF16: Validate types and call typed method
            if (output->native_type() != TensorType::BF16)
            {
                return false;
            }
            auto *bf16_output = dynamic_cast<BF16Tensor *>(output);
            if (!bf16_output)
            {
                return false;
            }
            return apply_bf16(embed_data, token_ids, num_tokens, d_model,
                              bf16_output->mutable_typed_data(), mpi_ctx, device_idx);
        }
        else if constexpr (std::is_same_v<TensorT, FP16Tensor>)
        {
            // FP16: Validate types and call typed method
            if (output->native_type() != TensorType::FP16)
            {
                return false;
            }
            auto *fp16_output = dynamic_cast<FP16Tensor *>(output);
            if (!fp16_output)
            {
                return false;
            }
            return apply_fp16(embed_data, token_ids, num_tokens, d_model,
                              fp16_output->mutable_typed_data(), mpi_ctx, device_idx);
        }
        else if constexpr (std::is_same_v<TensorT, Q8_1Tensor>)
        {
            // Q8_1: Validate types and call typed method
            if (output->native_type() != TensorType::Q8_1)
            {
                return false;
            }
            auto *q8_output = dynamic_cast<Q8_1Tensor *>(output);
            if (!q8_output)
            {
                return false;
            }
            return apply_q8_1(embed_data, token_ids, num_tokens, d_model,
                              q8_output->mutable_typed_data(), mpi_ctx, device_idx);
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
