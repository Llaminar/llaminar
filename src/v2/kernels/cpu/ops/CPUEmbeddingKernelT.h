/**
 * @file CPUEmbeddingKernelT.h
 * @brief CPU implementation of embedding lookup kernel
 *
 * Handles embedding table lookup with typed output:
 * - FP32 output: Direct memcpy from embedding table
 * - BF16/FP16 output: Lookup FP32 embeddings, then convert
 * - Q8_1 output: Lookup FP32 embeddings, then quantize to Q8_1 format
 *
 * The embedding table is always FP32 (stored in model weights).
 * Output format is determined by activation precision configuration.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include "../CPUKernelBase.h"
#include "../../common/EmbedQ8Repack.h"
#include <memory>

namespace llaminar2
{

    /**
     * @brief CPU implementation of embedding kernel
     *
     * @tparam TensorT The output tensor type (FP32Tensor, BF16Tensor, FP16Tensor, Q8_1Tensor)
     */
    template <typename TensorT>
    class CPUEmbeddingKernelT : public ITensorEmbedding, public CPUKernelBase
    {
    public:
        CPUEmbeddingKernelT() = default;
        ~CPUEmbeddingKernelT() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only
        }

        /**
         * @brief Execute embedding lookup with FP32 output
         */
        bool apply(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            float *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1);

        /**
         * @brief Execute embedding lookup with BF16 output
         */
        bool apply_bf16(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            uint16_t *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1);

        /**
         * @brief Execute embedding lookup with FP16 output
         */
        bool apply_fp16(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            uint16_t *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1);

        /**
         * @brief Execute embedding lookup with Q8_1 output
         */
        bool apply_q8_1(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            void *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1);

        /**
         * @brief Apply embedding lookup using tensor objects with automatic type dispatch
         */
        bool apply_tensor(
            const TensorBase *embed_table,
            const int *token_ids,
            int num_tokens,
            int d_model,
            TensorBase *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        KernelSnapshotInfo getKernelSnapshotInfo() const override
        {
            return KernelSnapshotInfo::embedding()
                .withWeight("embed_table", "embedding table [vocab_size, d_model]", KernelBufferDtype::FP32)
                .withInput("token_ids", "input token IDs [num_tokens]", KernelBufferDtype::INT32)
                .withOutput("output", "embedded output [num_tokens, d_model]", KernelBufferDtype::FP32)
                .withScalar("num_tokens", "number of tokens", KernelBufferDtype::INT32)
                .withScalar("d_model", "embedding dimension", KernelBufferDtype::INT32);
        }

    private:
        /// Cached EmbedQ8 repack of quantized embedding table (avoids full FP32 dequant)
        mutable const TensorBase *cached_embed_table_ = nullptr;
        mutable std::shared_ptr<const EmbedQ8RepackResult> cached_repack_;
    };

    // Backward compatibility alias
    using CPUEmbeddingKernel = CPUEmbeddingKernelT<FP32Tensor>;

} // namespace llaminar2
