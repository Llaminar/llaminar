/**
 * @file CUDAEmbeddingKernelT.h
 * @brief CUDA implementation of embedding lookup kernel
 *
 * Handles embedding table lookup on GPU. The embedding table is
 * uploaded to GPU memory and rows are looked up based on token IDs.
 *
 * Supports FP32 embedding tables with FP32 output.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"

namespace llaminar2
{

    /**
     * @brief CUDA implementation of embedding kernel for FP32
     */
    class CUDAEmbeddingKernelT : public ITensorEmbedding
    {
    public:
        explicit CUDAEmbeddingKernelT(int device_idx = 0) : device_idx_(device_idx) {}
        ~CUDAEmbeddingKernelT() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx >= 0; // GPU only
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
            int device_idx = 0) override;

        /**
         * @brief Execute embedding lookup with BF16 output (not yet implemented)
         */
        bool apply_bf16(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            uint16_t *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = 0) override;

        /**
         * @brief Execute embedding lookup with FP16 output (not yet implemented)
         */
        bool apply_fp16(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            uint16_t *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = 0) override;

        /**
         * @brief Execute embedding lookup with Q8_1 output (not yet implemented)
         */
        bool apply_q8_1(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            void *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = 0) override;

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
            int device_idx = 0) override;

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
        int device_idx_ = 0;
    };

    // Convenience alias
    using CUDAEmbeddingKernel = CUDAEmbeddingKernelT;

} // namespace llaminar2
