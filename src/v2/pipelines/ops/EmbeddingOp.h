/**
 * @file EmbeddingOp.h
 * @brief Typed self-validating embedding lookup operation
 *
 * Template-based operation supporting all activation precisions (FP32, BF16, FP16, Q8_1).
 * Uses compile-time dispatch via ActivationPrecision enum for zero-overhead precision handling.
 *
 * Handles embedding table lookup for transformer input processing:
 * - FP32 output: Direct memcpy from embedding table
 * - BF16/FP16 output: Lookup FP32 embeddings, then convert
 * - Q8_1 output: Lookup FP32 embeddings, then quantize to Q8_1 format
 *
 * The embedding table is always FP32 (stored in model weights).
 * Output format is determined by activation precision configuration.
 *
 * Usage:
 * @code
 * // Create typed op at initialization
 * auto embedding = createEmbeddingOp(ActivationPrecision::Q8_1);
 *
 * // Execute (polymorphic call, type-specific implementation)
 * TRY_OP(embedding->execute(embed_table, token_ids.data(), num_tokens, d_model, output, nullptr, device));
 * @endcode
 *
 * @author David Sanftenberg
 */

#pragma once

#include "Op.h"
#include "GemmOp.h" // For detail::PrecisionTensor
#include "../RuntimeConfig.h"
#include "../../kernels/cpu/ops/CPUEmbeddingKernelT.h"
#include "../../tensors/Tensors.h"
#include <memory>
#include <vector>
#include <cstring>

namespace llaminar2
{

    // =========================================================================
    // IEmbeddingOp Interface
    // =========================================================================

    /**
     * @brief Interface for typed embedding operations
     *
     * Provides polymorphic access to precision-specific embedding implementations.
     * Use createEmbeddingOp() factory to create instances at runtime.
     */
    class IEmbeddingOp
    {
    public:
        virtual ~IEmbeddingOp() = default;

        /**
         * @brief Get the activation precision this op was created for
         */
        virtual ActivationPrecision precision() const = 0;

        /**
         * @brief Execute embedding lookup
         *
         * @param embed_table Embedding table tensor [vocab_size, d_model] (always FP32)
         * @param token_ids Token IDs to look up
         * @param num_tokens Number of tokens
         * @param d_model Embedding dimension
         * @param output Output tensor [num_tokens, d_model]
         * @param mpi_ctx MPI context (nullptr for single-node)
         * @param device_idx Device index (-1 for CPU)
         *
         * @return true on success, false on failure
         */
        virtual bool execute(
            const TensorBase *embed_table,
            const int *token_ids,
            int num_tokens,
            int d_model,
            TensorBase *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;

        /**
         * @brief Execute batched embedding lookup (convenience overload)
         *
         * @param embed_table Embedding table tensor [vocab_size, d_model] (always FP32)
         * @param token_batches Vector of token sequences, one per batch element
         * @param padded_seq_len Padded sequence length (all sequences padded to this)
         * @param d_model Embedding dimension
         * @param output Output tensor [batch_size * padded_seq_len, d_model]
         * @param mpi_ctx MPI context (nullptr for single-node)
         * @param device_idx Device index (-1 for CPU)
         *
         * @return true on success, false on failure
         */
        virtual bool execute_batched(
            const TensorBase *embed_table,
            const std::vector<std::vector<int>> &token_batches,
            int padded_seq_len,
            int d_model,
            TensorBase *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;
    };

    // =========================================================================
    // EmbeddingOpTyped<Precision> Template Implementation
    // =========================================================================

    /**
     * @brief Typed embedding operation with compile-time precision dispatch
     *
     * @tparam Precision Activation precision (FP32, BF16, FP16, Q8_1)
     */
    template <ActivationPrecision Precision>
    class EmbeddingOpTyped : public IEmbeddingOp, public OpBase
    {
    public:
        using TensorT = typename detail::PrecisionTensor<Precision>::Type;
        using ElementType = typename detail::PrecisionTensor<Precision>::ElementType;

        const char *name() const override { return "EmbeddingOpTyped"; }
        ActivationPrecision precision() const override { return Precision; }

        /**
         * @brief Get the expected TensorType for this precision
         */
        static constexpr TensorType expected_tensor_type()
        {
            if constexpr (Precision == ActivationPrecision::FP32)
                return TensorType::FP32;
            else if constexpr (Precision == ActivationPrecision::BF16)
                return TensorType::BF16;
            else if constexpr (Precision == ActivationPrecision::FP16)
                return TensorType::FP16;
            else if constexpr (Precision == ActivationPrecision::Q8_1)
                return TensorType::Q8_1;
            else
                return TensorType::FP32;
        }

        bool execute(
            const TensorBase *embed_table,
            const int *token_ids,
            int num_tokens,
            int d_model,
            TensorBase *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            // Compile-time validation: ensure TensorT trait is defined
            static_assert(sizeof(TensorT) > 0,
                          "PrecisionTensor trait must be defined for this ActivationPrecision");

            // 1. Validate inputs
            if (!validateTensor(embed_table, "embed_table"))
                return false;
            if (!validateTensor(output, "output"))
                return false;
            if (!token_ids)
            {
                logError("token_ids is nullptr");
                return false;
            }
            if (num_tokens <= 0)
            {
                logError("num_tokens must be positive");
                return false;
            }
            if (d_model <= 0)
            {
                logError("d_model must be positive");
                return false;
            }

            // 2. Type validation: ensure output matches expected precision
            constexpr TensorType expected = expected_tensor_type();
            if (output->native_type() != expected)
            {
                logError(("output tensor type mismatch: expected " +
                          std::to_string(static_cast<int>(expected)) + ", got " +
                          std::to_string(static_cast<int>(output->native_type())))
                             .c_str());
                return false;
            }

            // 3. Create kernel from output tensor
            auto *activation = dynamic_cast<IActivationTensor *>(output);
            if (!activation)
            {
                logError("output tensor must be IActivationTensor");
                return false;
            }

            auto kernel = activation->createEmbedding();
            if (!kernel)
            {
                logError("failed to create Embedding kernel");
                return false;
            }

            // 4. Delegate to kernel's apply_tensor - handles all type dispatch internally
            if (!kernel->apply_tensor(embed_table, token_ids, num_tokens, d_model,
                                      output, mpi_ctx, device_idx))
            {
                logError("kernel execution failed");
                return false;
            }

            return true;
        }

        bool execute_batched(
            const TensorBase *embed_table,
            const std::vector<std::vector<int>> &token_batches,
            int padded_seq_len,
            int d_model,
            TensorBase *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            // 1. Validate inputs
            if (!validateTensor(embed_table, "embed_table"))
                return false;
            if (!validateTensor(output, "output"))
                return false;

            const int batch_size = static_cast<int>(token_batches.size());
            if (batch_size == 0)
            {
                logError("empty token_batches");
                return false;
            }

            // 2. Flatten token batches into a single buffer with padding
            const int total_positions = batch_size * padded_seq_len;
            std::vector<int> flat_token_ids(total_positions, 0); // Zero-initialized for padding

            int global_idx = 0;
            for (int b = 0; b < batch_size; ++b)
            {
                const auto &tokens = token_batches[b];
                const int seq_len = static_cast<int>(tokens.size());

                // Copy actual tokens
                for (int i = 0; i < seq_len && i < padded_seq_len; ++i)
                {
                    flat_token_ids[global_idx + i] = tokens[i];
                }
                // Padding positions are already zero (will lookup token 0, then need special handling)
                // For proper padding, we should mark these positions and zero them after
                global_idx += padded_seq_len;
            }

            // 3. Execute single embedding lookup call
            if (!execute(embed_table, flat_token_ids.data(), total_positions, d_model,
                         output, mpi_ctx, device_idx))
            {
                return false;
            }

            // 4. Zero out padding positions
            // This is needed because token_id=0 would lookup actual embeddings
            // For proper handling, we zero the output for padding positions
            global_idx = 0;
            for (int b = 0; b < batch_size; ++b)
            {
                const auto &tokens = token_batches[b];
                const int seq_len = static_cast<int>(tokens.size());

                // Zero out padding positions
                for (int i = seq_len; i < padded_seq_len; ++i)
                {
                    zero_output_row(output, global_idx + i, d_model);
                }
                global_idx += padded_seq_len;
            }

            return true;
        }

    private:
        /**
         * @brief Zero out a single row in the output tensor
         */
        void zero_output_row(TensorBase *output, int row_idx, int d_model)
        {
            if constexpr (Precision == ActivationPrecision::FP32)
            {
                float *out_data = output->mutable_data();
                if (out_data)
                {
                    std::memset(out_data + row_idx * d_model, 0, d_model * sizeof(float));
                }
            }
            else if constexpr (Precision == ActivationPrecision::BF16)
            {
                auto *bf16_output = dynamic_cast<BF16Tensor *>(output);
                if (bf16_output)
                {
                    uint16_t *out_data = bf16_output->mutable_bf16_data();
                    if (out_data)
                    {
                        std::memset(out_data + row_idx * d_model, 0, d_model * sizeof(uint16_t));
                    }
                }
            }
            else if constexpr (Precision == ActivationPrecision::FP16)
            {
                auto *fp16_output = dynamic_cast<FP16Tensor *>(output);
                if (fp16_output)
                {
                    uint16_t *out_data = fp16_output->mutable_fp16_data();
                    if (out_data)
                    {
                        std::memset(out_data + row_idx * d_model, 0, d_model * sizeof(uint16_t));
                    }
                }
            }
            else if constexpr (Precision == ActivationPrecision::Q8_1)
            {
                auto *q8_output = dynamic_cast<Q8_1Tensor *>(output);
                if (q8_output)
                {
                    const int blocks_per_row = d_model / 32;
                    Q8_1Block *out_blocks = q8_output->mutable_q8_1_blocks();
                    if (out_blocks)
                    {
                        Q8_1Block *row_blocks = out_blocks + row_idx * blocks_per_row;
                        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx)
                        {
                            Q8_1Block &block = row_blocks[block_idx];
                            block.d = 0;
                            block.sum_qs = 0;
                            std::memset(block.qs, 0, 32);
                        }
                    }
                }
            }
        }
    };

    // =========================================================================
    // Factory Function
    // =========================================================================

    /**
     * @brief Create typed embedding op for given precision
     *
     * @param precision Activation precision
     * @return Unique pointer to precision-specific embedding op
     */
    inline std::unique_ptr<IEmbeddingOp> createEmbeddingOp(ActivationPrecision precision)
    {
        switch (precision)
        {
        case ActivationPrecision::Q8_1:
            return std::make_unique<EmbeddingOpTyped<ActivationPrecision::Q8_1>>();
        case ActivationPrecision::BF16:
            return std::make_unique<EmbeddingOpTyped<ActivationPrecision::BF16>>();
        case ActivationPrecision::FP16:
            return std::make_unique<EmbeddingOpTyped<ActivationPrecision::FP16>>();
        case ActivationPrecision::FP32:
        default:
            return std::make_unique<EmbeddingOpTyped<ActivationPrecision::FP32>>();
        }
    }

    // =========================================================================
    // Legacy Compatibility (deprecated)
    // =========================================================================

    /**
     * @brief Legacy EmbeddingOp for backward compatibility
     * @deprecated Use createEmbeddingOp() factory instead
     */
    class EmbeddingOp : public OpBase
    {
    public:
        const char *name() const override { return "EmbeddingOp"; }

        /**
         * @brief Execute batched embedding lookup (legacy interface)
         */
        bool operator()(
            const TensorBase *embed_table,
            const std::vector<std::vector<int>> &token_batches,
            int padded_seq_len,
            int d_model,
            TensorBase *output)
        {
            // Determine precision from output tensor type
            ActivationPrecision precision;
            switch (output->native_type())
            {
            case TensorType::FP32:
                precision = ActivationPrecision::FP32;
                break;
            case TensorType::BF16:
                precision = ActivationPrecision::BF16;
                break;
            case TensorType::FP16:
                precision = ActivationPrecision::FP16;
                break;
            case TensorType::Q8_1:
                precision = ActivationPrecision::Q8_1;
                break;
            default:
                logError(("unsupported output tensor type: " +
                          std::to_string(static_cast<int>(output->native_type())))
                             .c_str());
                return false;
            }

            auto op = createEmbeddingOp(precision);
            return op->execute_batched(embed_table, token_batches, padded_seq_len, d_model,
                                       output, nullptr, -1);
        }
    };

} // namespace llaminar2
