/**
 * @file ResidualOp.h
 * @brief Typed residual connection operations for transformer layers
 * @author David Sanftenberg
 *
 * Provides precision-aware residual addition (hidden = residual + projection)
 * with native paths for FP32, BF16, FP16, and Q8_1 activation precisions.
 */

#pragma once

#include "Op.h"
#include "../PipelineConfig.h"
#include "../../tensors/Tensors.h"
#include "../../tensors/SIMDHelpers.h"
#include "../../utils/DebugEnv.h"
#include "../../utils/Logger.h"
#include "../../utils/OpenMPUtils.h"

#include <cmath>
#include <memory>
#include <omp.h>

namespace llaminar2
{

    // ============================================================================
    // IResidualOp Interface
    // ============================================================================

    /**
     * @brief Abstract interface for residual operations
     *
     * Residual connections add projection outputs back to the residual stream:
     *   hidden = residual + projection
     *
     * This is used in transformer blocks after attention and FFN layers.
     */
    class IResidualOp
    {
    public:
        virtual ~IResidualOp() = default;

        /**
         * @brief Apply residual connection
         * @param residual Input residual tensor (read)
         * @param projection Projection tensor to add (read)
         * @param hidden Output tensor (write)
         * @param effective_seq_len Sequence length
         * @param d_model Model dimension
         * @return true on success
         */
        virtual bool apply(
            TensorBase *residual,
            TensorBase *projection,
            TensorBase *hidden,
            int effective_seq_len,
            int d_model) = 0;

        /**
         * @brief Apply batched residual connection with padding mask
         * @param residual Input residual tensor (read)
         * @param projection Projection tensor to add (read)
         * @param hidden Output tensor (write)
         * @param batch_size Number of sequences in batch
         * @param seq_lengths Array of actual sequence lengths per batch
         * @param max_seq_len Maximum sequence length (padded)
         * @param d_model Model dimension
         * @return true on success
         */
        virtual bool batched(
            TensorBase *residual,
            TensorBase *projection,
            TensorBase *hidden,
            int batch_size,
            const int *seq_lengths,
            int max_seq_len,
            int d_model) = 0;

        /**
         * @brief Get the activation precision this op handles
         */
        virtual ActivationPrecision precision() const = 0;
    };

    // ============================================================================
    // ResidualOpTyped<P> Template Implementation
    // ============================================================================

    namespace detail
    {

        /**
         * @brief Map ActivationPrecision to tensor type for residual ops
         */
        template <ActivationPrecision P>
        struct ResidualTensor;

        template <>
        struct ResidualTensor<ActivationPrecision::FP32>
        {
            using type = FP32Tensor;
        };

        template <>
        struct ResidualTensor<ActivationPrecision::BF16>
        {
            using type = BF16Tensor;
        };

        template <>
        struct ResidualTensor<ActivationPrecision::FP16>
        {
            using type = FP16Tensor;
        };

        template <>
        struct ResidualTensor<ActivationPrecision::Q8_1>
        {
            using type = Q8_1Tensor;
        };

    } // namespace detail

    /**
     * @brief Typed residual operation implementation
     *
     * Template specializations provide precision-specific implementations:
     * - FP32: Standard float addition with OpenMP
     * - BF16: BF16 addition (converts to FP32 internally)
     * - FP16: FP16 addition (converts to FP32 internally)
     * - Q8_1: Native quantized addition using SIMD intrinsics
     */
    template <ActivationPrecision P>
    class ResidualOpTyped : public IResidualOp, public OpBase
    {
    public:
        using TensorType = typename detail::ResidualTensor<P>::type;

        ResidualOpTyped() = default;

        const char *name() const override { return "ResidualOpTyped"; }
        ActivationPrecision precision() const override { return P; }

        bool apply(
            TensorBase *residual,
            TensorBase *projection,
            TensorBase *hidden,
            int effective_seq_len,
            int d_model) override
        {
            return applyImpl(residual, projection, hidden, effective_seq_len, d_model);
        }

        bool batched(
            TensorBase *residual,
            TensorBase *projection,
            TensorBase *hidden,
            int batch_size,
            const int *seq_lengths,
            int max_seq_len,
            int d_model) override
        {
            return batchedImpl(residual, projection, hidden, batch_size, seq_lengths, max_seq_len, d_model);
        }

    private:
        // Primary template - FP32 implementation
        bool applyImpl(
            TensorBase *residual,
            TensorBase *projection,
            TensorBase *hidden,
            int effective_seq_len,
            int d_model)
        {
            auto *res_typed = dynamic_cast<TensorType *>(residual);
            auto *proj_typed = dynamic_cast<TensorType *>(projection);
            auto *out_typed = dynamic_cast<TensorType *>(hidden);

            if (!res_typed || !proj_typed || !out_typed)
            {
                LOG_ERROR("ResidualOpTyped<" << static_cast<int>(P) << ">: tensor type mismatch");
                return false;
            }

            const size_t n = static_cast<size_t>(effective_seq_len) * static_cast<size_t>(d_model);

            if constexpr (P == ActivationPrecision::FP32)
            {
                const float *r = res_typed->data();
                const float *p = proj_typed->data();
                float *h = out_typed->mutable_data();

                auto residual_work = [&]()
                {
#pragma omp for schedule(static)
                    for (size_t i = 0; i < n; ++i)
                    {
                        h[i] = r[i] + p[i];
                    }
                };
                OMP_WORKSHARE_REGION(residual_work);
            }
            else if constexpr (P == ActivationPrecision::BF16)
            {
                // Use native BF16 accessors (not dequantized data())
                const uint16_t *r = res_typed->bf16_data();
                const uint16_t *p = proj_typed->bf16_data();
                uint16_t *h = out_typed->mutable_bf16_data();

                auto residual_work = [&]()
                {
#pragma omp for schedule(static)
                    for (size_t i = 0; i < n; ++i)
                    {
                        float rf = simd::bf16_to_fp32(r[i]);
                        float pf = simd::bf16_to_fp32(p[i]);
                        h[i] = simd::fp32_to_bf16(rf + pf);
                    }
                };
                OMP_WORKSHARE_REGION(residual_work);
            }
            else if constexpr (P == ActivationPrecision::FP16)
            {
                // Use native FP16 accessors (not dequantized data())
                const uint16_t *r = res_typed->fp16_data();
                const uint16_t *p = proj_typed->fp16_data();
                uint16_t *h = out_typed->mutable_fp16_data();

                auto residual_work = [&]()
                {
#pragma omp for schedule(static)
                    for (size_t i = 0; i < n; ++i)
                    {
                        float rf = simd::fp16_to_fp32(r[i]);
                        float pf = simd::fp16_to_fp32(p[i]);
                        h[i] = simd::fp32_to_fp16(rf + pf);
                    }
                };
                OMP_WORKSHARE_REGION(residual_work);
            }
            else if constexpr (P == ActivationPrecision::Q8_1)
            {
                // Native Q8_1 addition using SIMD
                return applyQ8_1(res_typed, proj_typed, out_typed, effective_seq_len, d_model);
            }

            return true;
        }

        bool batchedImpl(
            TensorBase *residual,
            TensorBase *projection,
            TensorBase *hidden,
            int batch_size,
            const int *seq_lengths,
            int max_seq_len,
            int d_model)
        {
            auto *res_typed = dynamic_cast<TensorType *>(residual);
            auto *proj_typed = dynamic_cast<TensorType *>(projection);
            auto *out_typed = dynamic_cast<TensorType *>(hidden);

            if (!res_typed || !proj_typed || !out_typed)
            {
                LOG_ERROR("ResidualOpTyped<" << static_cast<int>(P) << ">::batched: tensor type mismatch");
                return false;
            }

            if constexpr (P == ActivationPrecision::FP32)
            {
                const float *r = res_typed->data();
                const float *p = proj_typed->data();
                float *h = out_typed->mutable_data();

                auto batched_work = [&]()
                {
#pragma omp for collapse(2) schedule(static)
                    for (int b = 0; b < batch_size; ++b)
                    {
                        for (int s = 0; s < max_seq_len; ++s)
                        {
                            const size_t row_offset = (static_cast<size_t>(b) * max_seq_len + s) * d_model;
                            if (s < seq_lengths[b])
                            {
                                // Valid position: add residual + projection
                                for (int d = 0; d < d_model; ++d)
                                {
                                    h[row_offset + d] = r[row_offset + d] + p[row_offset + d];
                                }
                            }
                            else
                            {
                                // Padding position: zero out
                                for (int d = 0; d < d_model; ++d)
                                {
                                    h[row_offset + d] = 0.0f;
                                }
                            }
                        }
                    }
                };
                OMP_WORKSHARE_REGION(batched_work);
            }
            else if constexpr (P == ActivationPrecision::BF16)
            {
                // Use native BF16 accessors (not dequantized data())
                const uint16_t *r = res_typed->bf16_data();
                const uint16_t *p = proj_typed->bf16_data();
                uint16_t *h = out_typed->mutable_bf16_data();

                auto batched_work = [&]()
                {
#pragma omp for collapse(2) schedule(static)
                    for (int b = 0; b < batch_size; ++b)
                    {
                        for (int s = 0; s < max_seq_len; ++s)
                        {
                            const size_t row_offset = (static_cast<size_t>(b) * max_seq_len + s) * d_model;
                            if (s < seq_lengths[b])
                            {
                                for (int d = 0; d < d_model; ++d)
                                {
                                    float rf = simd::bf16_to_fp32(r[row_offset + d]);
                                    float pf = simd::bf16_to_fp32(p[row_offset + d]);
                                    h[row_offset + d] = simd::fp32_to_bf16(rf + pf);
                                }
                            }
                            else
                            {
                                for (int d = 0; d < d_model; ++d)
                                {
                                    h[row_offset + d] = 0;
                                }
                            }
                        }
                    }
                };
                OMP_WORKSHARE_REGION(batched_work);
            }
            else if constexpr (P == ActivationPrecision::FP16)
            {
                // Use native FP16 accessors (not dequantized data())
                const uint16_t *r = res_typed->fp16_data();
                const uint16_t *p = proj_typed->fp16_data();
                uint16_t *h = out_typed->mutable_fp16_data();

                auto batched_work = [&]()
                {
#pragma omp for collapse(2) schedule(static)
                    for (int b = 0; b < batch_size; ++b)
                    {
                        for (int s = 0; s < max_seq_len; ++s)
                        {
                            const size_t row_offset = (static_cast<size_t>(b) * max_seq_len + s) * d_model;
                            if (s < seq_lengths[b])
                            {
                                for (int d = 0; d < d_model; ++d)
                                {
                                    float rf = simd::fp16_to_fp32(r[row_offset + d]);
                                    float pf = simd::fp16_to_fp32(p[row_offset + d]);
                                    h[row_offset + d] = simd::fp32_to_fp16(rf + pf);
                                }
                            }
                            else
                            {
                                for (int d = 0; d < d_model; ++d)
                                {
                                    h[row_offset + d] = 0;
                                }
                            }
                        }
                    }
                };
                OMP_WORKSHARE_REGION(batched_work);
            }
            else if constexpr (P == ActivationPrecision::Q8_1)
            {
                return batchedQ8_1(res_typed, proj_typed, out_typed, batch_size, seq_lengths, max_seq_len, d_model);
            }

            return true;
        }

        // Q8_1-specific implementations
        bool applyQ8_1(
            Q8_1Tensor *residual,
            Q8_1Tensor *projection,
            Q8_1Tensor *hidden,
            int effective_seq_len,
            int d_model)
        {
            const size_t total_elements = static_cast<size_t>(effective_seq_len) * d_model;
            constexpr size_t BLOCK_SIZE = Q8_1Block::BLOCK_SIZE;
            const size_t num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // Use native Q8_1 accessors
            const Q8_1Block *res_blocks = residual->q8_1_blocks();
            const Q8_1Block *proj_blocks = projection->q8_1_blocks();
            Q8_1Block *out_blocks = hidden->mutable_q8_1_blocks();

            // Use SIMD Q8_1 addition - process all blocks at once
            // q8_1_add_q8_1 expects element count, not block count
            simd::q8_1_add_q8_1(
                res_blocks,
                proj_blocks,
                out_blocks,
                num_blocks * BLOCK_SIZE // element count
            );

            return true;
        }

        bool batchedQ8_1(
            Q8_1Tensor *residual,
            Q8_1Tensor *projection,
            Q8_1Tensor *hidden,
            int batch_size,
            const int *seq_lengths,
            int max_seq_len,
            int d_model)
        {
            constexpr size_t BLOCK_SIZE = Q8_1Block::BLOCK_SIZE;
            const size_t blocks_per_row = (d_model + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // Use native Q8_1 accessors
            const Q8_1Block *res_blocks = residual->q8_1_blocks();
            const Q8_1Block *proj_blocks = projection->q8_1_blocks();
            Q8_1Block *out_blocks = hidden->mutable_q8_1_blocks();

            auto batched_q8_1_work = [&]()
            {
#pragma omp for collapse(2) schedule(static)
                for (int b = 0; b < batch_size; ++b)
                {
                    for (int s = 0; s < max_seq_len; ++s)
                    {
                        const size_t row_idx = static_cast<size_t>(b) * max_seq_len + s;
                        const size_t block_row_offset = row_idx * blocks_per_row;

                        if (s < seq_lengths[b])
                        {
                            // Valid position: add blocks using SIMD
                            // Process one row at a time (blocks_per_row blocks = blocks_per_row * 32 elements)
                            simd::q8_1_add_q8_1(
                                &res_blocks[block_row_offset],
                                &proj_blocks[block_row_offset],
                                &out_blocks[block_row_offset],
                                blocks_per_row * BLOCK_SIZE // element count for this row
                            );
                        }
                        else
                        {
                            // Padding position: zero out blocks
                            for (size_t blk = 0; blk < blocks_per_row; ++blk)
                            {
                                Q8_1Block &out_blk = out_blocks[block_row_offset + blk];
                                out_blk.d = 0;
                                out_blk.sum_qs = 0;
                                std::memset(out_blk.qs, 0, BLOCK_SIZE);
                            }
                        }
                    }
                }
            };
            OMP_WORKSHARE_REGION(batched_q8_1_work);

            return true;
        }

        // BF16 conversion helpers
        static float bf16_to_fp32(uint16_t bf16)
        {
            uint32_t fp32_bits = static_cast<uint32_t>(bf16) << 16;
            float result;
            std::memcpy(&result, &fp32_bits, sizeof(float));
            return result;
        }

        static uint16_t fp32_to_bf16(float fp32)
        {
            uint32_t fp32_bits;
            std::memcpy(&fp32_bits, &fp32, sizeof(uint32_t));
            // Round to nearest even
            uint32_t rounding_bias = ((fp32_bits >> 16) & 1) + 0x7FFF;
            return static_cast<uint16_t>((fp32_bits + rounding_bias) >> 16);
        }

        // FP16 conversion helpers
        static float fp16_to_fp32(uint16_t fp16)
        {
            uint32_t sign = (fp16 >> 15) & 0x1;
            uint32_t exp = (fp16 >> 10) & 0x1F;
            uint32_t mant = fp16 & 0x3FF;

            uint32_t fp32_bits;
            if (exp == 0)
            {
                if (mant == 0)
                {
                    fp32_bits = sign << 31;
                }
                else
                {
                    // Denormalized
                    exp = 1;
                    while ((mant & 0x400) == 0)
                    {
                        mant <<= 1;
                        exp--;
                    }
                    mant &= 0x3FF;
                    fp32_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                }
            }
            else if (exp == 31)
            {
                fp32_bits = (sign << 31) | 0x7F800000 | (mant << 13);
            }
            else
            {
                fp32_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
            }

            float result;
            std::memcpy(&result, &fp32_bits, sizeof(float));
            return result;
        }

        static uint16_t fp32_to_fp16(float fp32)
        {
            uint32_t fp32_bits;
            std::memcpy(&fp32_bits, &fp32, sizeof(uint32_t));

            uint32_t sign = (fp32_bits >> 31) & 0x1;
            int32_t exp = ((fp32_bits >> 23) & 0xFF) - 127 + 15;
            uint32_t mant = (fp32_bits >> 13) & 0x3FF;

            if (exp <= 0)
            {
                return static_cast<uint16_t>(sign << 15);
            }
            else if (exp >= 31)
            {
                return static_cast<uint16_t>((sign << 15) | 0x7C00);
            }

            return static_cast<uint16_t>((sign << 15) | (exp << 10) | mant);
        }
    };

    // ============================================================================
    // Factory Function
    // ============================================================================

    /**
     * @brief Create a typed residual op for the given activation precision
     * @param precision The activation precision to use
     * @return Unique pointer to IResidualOp implementation
     */
    inline std::unique_ptr<IResidualOp> createResidualOp(ActivationPrecision precision)
    {
        switch (precision)
        {
        case ActivationPrecision::FP32:
            return std::make_unique<ResidualOpTyped<ActivationPrecision::FP32>>();
        case ActivationPrecision::BF16:
            return std::make_unique<ResidualOpTyped<ActivationPrecision::BF16>>();
        case ActivationPrecision::FP16:
            return std::make_unique<ResidualOpTyped<ActivationPrecision::FP16>>();
        case ActivationPrecision::Q8_1:
            return std::make_unique<ResidualOpTyped<ActivationPrecision::Q8_1>>();
        default:
            LOG_ERROR("createResidualOp: unsupported precision " << static_cast<int>(precision));
            return nullptr;
        }
    }

} // namespace llaminar2
