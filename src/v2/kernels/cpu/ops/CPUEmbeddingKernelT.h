/**
 * @file CPUEmbeddingKernelT.h
 * @brief CPU implementation of embedding lookup kernel
 *
 * Handles embedding table lookup with typed output:
 * - FP32 output: Direct memcpy from embedding table
 * - BF16/FP16 output: Lookup FP32 embeddings, then convert
 * - Q8_1 output: Lookup FP32 embeddings, then quantize to Q8_1 format
 *
 * Native quantized weights retain their original codebook and scale metadata.
 * Only consumed rows are decoded; no derived whole-table quantization is cached.
 * Output format is determined by activation precision configuration.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include "../CPUKernelBase.h"
#include "../../common/NativeEmbeddingLayout.h"
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
            const IMPIContext *mpi_ctx = nullptr,
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
            const IMPIContext *mpi_ctx = nullptr,
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
            const IMPIContext *mpi_ctx = nullptr,
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
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1);

        /**
         * @brief Decode requested native weight rows and publish the declared activation type.
         * @param embed_table Live native source table; quantized weights are never requantized.
         * @param token_ids Host-owned global vocabulary IDs for this CPU transaction.
         * @param num_tokens Number of live requested rows.
         * @param d_model Logical values per source and output row.
         * @param output Caller-owned FP32/BF16/FP16/Q8_1 activation storage.
         * @param mpi_ctx Optional declared vocabulary-shard context.
         * @param device_idx CPU execution identity, retained by the common interface.
         * @return True after every live output row has been written, false for invalid input.
         * @throws std::runtime_error If a released or malformed source cannot decode its row.
         */
        bool apply_tensor(
            const TensorBase *embed_table,
            const int *token_ids,
            int num_tokens,
            int d_model,
            TensorBase *output,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        void setVocabRange(int vocab_offset, int local_vocab_size) override
        {
            explicit_vocab_range_ = true;
            vocab_offset_ = vocab_offset;
            local_vocab_size_ = local_vocab_size;
        }

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
        /// Vocab-parallel sharding state (set by apply_tensor, used by inner apply* functions)
        /// When embedding table is sharded across TP ranks, each rank holds a slice of the
        /// vocabulary. vocab_offset_ is the first global vocab index in this shard.
        /// local_vocab_size_ is the number of rows in this shard.
        /// Tokens outside [vocab_offset_, vocab_offset_ + local_vocab_size_) produce zeros.
        int vocab_offset_ = 0;
        int local_vocab_size_ = 0;
        bool explicit_vocab_range_ = false;
        bool vocab_parallel_ = false;
    };

    // Backward compatibility alias
    using CPUEmbeddingKernel = CPUEmbeddingKernelT<FP32Tensor>;

} // namespace llaminar2
