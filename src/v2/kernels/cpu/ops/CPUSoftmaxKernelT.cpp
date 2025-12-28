/**
 * @file CPUSoftmaxKernelT.cpp
 * @brief Implementation of typed Softmax kernel with ActivationPrecision support
 * @author David Sanftenberg
 *
 * This file implements the CPUSoftmaxKernelT template specializations for
 * FP32, BF16, FP16, and Q8_1 precision types.
 *
 * Each precision uses native primitives without unnecessary type conversions:
 * - FP32: Direct FP32 softmax
 * - BF16: Native BF16 softmax (converts to FP32 only for exp/div)
 * - FP16: Native FP16 softmax (converts to FP32 only for exp/div)
 * - Q8_1: Integer-aware softmax (minimizes FP32 usage, direct requantization)
 */

#include "CPUSoftmaxKernelT.h"
#include "../primitives/SoftmaxPrimitives_New.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"

namespace llaminar2
{

    // ============================================================================
    // FP32 Specialization Implementation
    // ============================================================================

    bool CPUSoftmaxKernelT<ActivationPrecision::FP32>::apply_typed(
        float *data,
        int rows,
        int cols,
        bool use_causal_mask,
        float scale,
        int device_idx)
    {
        (void)device_idx; // Unused for CPU kernel

        if (!data)
        {
            LOG_ERROR("CPUSoftmaxKernelT<FP32>: Null data pointer");
            return false;
        }

        if (rows <= 0 || cols <= 0)
        {
            LOG_DEBUG("CPUSoftmaxKernelT<FP32>: Empty input (rows=" << rows << ", cols=" << cols << ")");
            return true; // Nothing to do
        }

        // Use row-major batch softmax primitive with OpenMP parallelization
        primitives::softmax_row_major_fp32(
            data,
            rows,
            cols,
            use_causal_mask,
            scale,
            true // parallel
        );

        return true;
    }

    // ============================================================================
    // BF16 Specialization Implementation
    // ============================================================================

    bool CPUSoftmaxKernelT<ActivationPrecision::BF16>::apply_typed(
        uint16_t *data,
        int rows,
        int cols,
        bool use_causal_mask,
        float scale,
        int device_idx)
    {
        (void)device_idx; // Unused for CPU kernel

        if (!data)
        {
            LOG_ERROR("CPUSoftmaxKernelT<BF16>: Null data pointer");
            return false;
        }

        if (rows <= 0 || cols <= 0)
        {
            LOG_DEBUG("CPUSoftmaxKernelT<BF16>: Empty input (rows=" << rows << ", cols=" << cols << ")");
            return true; // Nothing to do
        }

        // Use native BF16 softmax primitive
        // Internally converts to FP32 only for exp/div, accumulates in FP32,
        // then converts back to BF16 for output
        primitives::softmax_row_major_bf16(
            data,
            rows,
            cols,
            use_causal_mask,
            scale,
            true // parallel
        );

        return true;
    }

    // ============================================================================
    // FP16 Specialization Implementation
    // ============================================================================

    bool CPUSoftmaxKernelT<ActivationPrecision::FP16>::apply_typed(
        uint16_t *data,
        int rows,
        int cols,
        bool use_causal_mask,
        float scale,
        int device_idx)
    {
        (void)device_idx; // Unused for CPU kernel

        if (!data)
        {
            LOG_ERROR("CPUSoftmaxKernelT<FP16>: Null data pointer");
            return false;
        }

        if (rows <= 0 || cols <= 0)
        {
            LOG_DEBUG("CPUSoftmaxKernelT<FP16>: Empty input (rows=" << rows << ", cols=" << cols << ")");
            return true; // Nothing to do
        }

        // Use native FP16 softmax primitive
        // Internally converts to FP32 only for exp/div, accumulates in FP32,
        // then converts back to FP16 for output
        primitives::softmax_row_major_fp16(
            data,
            rows,
            cols,
            use_causal_mask,
            scale,
            true // parallel
        );

        return true;
    }

    // ============================================================================
    // Q8_1 Specialization Implementation (Integer-Aware)
    // ============================================================================

    bool CPUSoftmaxKernelT<ActivationPrecision::Q8_1>::apply_typed(
        Q8_1Block *data,
        int rows,
        int n_blocks_per_row,
        bool use_causal_mask,
        float scale,
        int device_idx)
    {
        (void)device_idx; // Unused for CPU kernel

        if (!data)
        {
            LOG_ERROR("CPUSoftmaxKernelT<Q8_1>: Null data pointer");
            return false;
        }

        if (rows <= 0 || n_blocks_per_row <= 0)
        {
            LOG_DEBUG("CPUSoftmaxKernelT<Q8_1>: Empty input (rows=" << rows
                                                                    << ", n_blocks_per_row=" << n_blocks_per_row << ")");
            return true; // Nothing to do
        }

        // Use integer-aware Q8_1 softmax primitive
        // This uses:
        //   - Integer max-finding in Q8_1 blocks (compare qs values)
        //   - Batched scale multiplication (1 mul per block instead of per element)
        //   - Direct requantization to Q8_1 output (no intermediate FP32 storage)
        primitives::softmax_row_major_q8_1(
            data,
            rows,
            n_blocks_per_row,
            use_causal_mask,
            scale,
            true // parallel
        );

        return true;
    }

    // ============================================================================
    // apply_tensor() Implementations (Polymorphic Interface)
    // ============================================================================

    bool CPUSoftmaxKernelT<ActivationPrecision::FP32>::apply_tensor(
        const TensorBase *input,
        TensorBase *output,
        int rows, int cols,
        bool use_causal_mask,
        float scale,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;

        if (!input || !output)
        {
            LOG_ERROR("CPUSoftmaxKernelT<FP32>::apply_tensor: Null tensor pointer");
            return false;
        }

        // Verify types match expected precision
        if (input->native_type() != TensorType::FP32 || output->native_type() != TensorType::FP32)
        {
            LOG_ERROR("CPUSoftmaxKernelT<FP32>::apply_tensor: Type mismatch - expected FP32");
            return false;
        }

        const float *in_data = input->data();
        float *out_data = output->mutable_data();

        if (!in_data || !out_data)
        {
            LOG_ERROR("CPUSoftmaxKernelT<FP32>::apply_tensor: Null data pointer");
            return false;
        }

        // Copy input to output for in-place operation if different buffers
        if (in_data != out_data)
        {
            std::memcpy(out_data, in_data, static_cast<size_t>(rows) * cols * sizeof(float));
        }

        return apply_typed(out_data, rows, cols, use_causal_mask, scale, device_idx);
    }

    bool CPUSoftmaxKernelT<ActivationPrecision::BF16>::apply_tensor(
        const TensorBase *input,
        TensorBase *output,
        int rows, int cols,
        bool use_causal_mask,
        float scale,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;

        if (!input || !output)
        {
            LOG_ERROR("CPUSoftmaxKernelT<BF16>::apply_tensor: Null tensor pointer");
            return false;
        }

        // Verify types match expected precision
        if (input->native_type() != TensorType::BF16 || output->native_type() != TensorType::BF16)
        {
            LOG_ERROR("CPUSoftmaxKernelT<BF16>::apply_tensor: Type mismatch - expected BF16");
            return false;
        }

        auto *bf16_input = dynamic_cast<const BF16Tensor *>(input);
        auto *bf16_output = dynamic_cast<BF16Tensor *>(output);
        if (!bf16_input || !bf16_output)
        {
            LOG_ERROR("CPUSoftmaxKernelT<BF16>::apply_tensor: Failed to cast to BF16Tensor");
            return false;
        }

        const uint16_t *in_data = bf16_input->typed_data();
        uint16_t *out_data = bf16_output->mutable_typed_data();

        if (!in_data || !out_data)
        {
            LOG_ERROR("CPUSoftmaxKernelT<BF16>::apply_tensor: Null data pointer");
            return false;
        }

        // Copy input to output for in-place operation if different buffers
        if (in_data != out_data)
        {
            std::memcpy(out_data, in_data, static_cast<size_t>(rows) * cols * sizeof(uint16_t));
        }

        return apply_typed(out_data, rows, cols, use_causal_mask, scale, device_idx);
    }

    bool CPUSoftmaxKernelT<ActivationPrecision::FP16>::apply_tensor(
        const TensorBase *input,
        TensorBase *output,
        int rows, int cols,
        bool use_causal_mask,
        float scale,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;

        if (!input || !output)
        {
            LOG_ERROR("CPUSoftmaxKernelT<FP16>::apply_tensor: Null tensor pointer");
            return false;
        }

        // Verify types match expected precision
        if (input->native_type() != TensorType::FP16 || output->native_type() != TensorType::FP16)
        {
            LOG_ERROR("CPUSoftmaxKernelT<FP16>::apply_tensor: Type mismatch - expected FP16");
            return false;
        }

        auto *fp16_input = dynamic_cast<const FP16Tensor *>(input);
        auto *fp16_output = dynamic_cast<FP16Tensor *>(output);
        if (!fp16_input || !fp16_output)
        {
            LOG_ERROR("CPUSoftmaxKernelT<FP16>::apply_tensor: Failed to cast to FP16Tensor");
            return false;
        }

        const uint16_t *in_data = fp16_input->typed_data();
        uint16_t *out_data = fp16_output->mutable_typed_data();

        if (!in_data || !out_data)
        {
            LOG_ERROR("CPUSoftmaxKernelT<FP16>::apply_tensor: Null data pointer");
            return false;
        }

        // Copy input to output for in-place operation if different buffers
        if (in_data != out_data)
        {
            std::memcpy(out_data, in_data, static_cast<size_t>(rows) * cols * sizeof(uint16_t));
        }

        return apply_typed(out_data, rows, cols, use_causal_mask, scale, device_idx);
    }

    bool CPUSoftmaxKernelT<ActivationPrecision::Q8_1>::apply_tensor(
        const TensorBase *input,
        TensorBase *output,
        int rows, int cols,
        bool use_causal_mask,
        float scale,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;

        if (!input || !output)
        {
            LOG_ERROR("CPUSoftmaxKernelT<Q8_1>::apply_tensor: Null tensor pointer");
            return false;
        }

        // Verify types match expected precision
        if (input->native_type() != TensorType::Q8_1 || output->native_type() != TensorType::Q8_1)
        {
            LOG_ERROR("CPUSoftmaxKernelT<Q8_1>::apply_tensor: Type mismatch - expected Q8_1");
            return false;
        }

        auto *q8_input = dynamic_cast<const Q8_1Tensor *>(input);
        auto *q8_output = dynamic_cast<Q8_1Tensor *>(output);
        if (!q8_input || !q8_output)
        {
            LOG_ERROR("CPUSoftmaxKernelT<Q8_1>::apply_tensor: Failed to cast to Q8_1Tensor");
            return false;
        }

        const Q8_1Block *in_blocks = q8_input->typed_data();
        Q8_1Block *out_blocks = q8_output->mutable_typed_data();

        if (!in_blocks || !out_blocks)
        {
            LOG_ERROR("CPUSoftmaxKernelT<Q8_1>::apply_tensor: Null block pointer");
            return false;
        }

        // cols here is n_blocks_per_row for Q8_1
        int n_blocks_per_row = cols;

        // Copy input to output for in-place operation if different buffers
        if (in_blocks != out_blocks)
        {
            std::memcpy(out_blocks, in_blocks, static_cast<size_t>(rows) * n_blocks_per_row * sizeof(Q8_1Block));
        }

        return apply_typed(out_blocks, rows, n_blocks_per_row, use_causal_mask, scale, device_idx);
    }

    // ============================================================================
    // Destructor Definitions (required for vtable emission)
    // ============================================================================

    CPUSoftmaxKernelT<ActivationPrecision::FP32>::~CPUSoftmaxKernelT() = default;
    CPUSoftmaxKernelT<ActivationPrecision::BF16>::~CPUSoftmaxKernelT() = default;
    CPUSoftmaxKernelT<ActivationPrecision::FP16>::~CPUSoftmaxKernelT() = default;
    CPUSoftmaxKernelT<ActivationPrecision::Q8_1>::~CPUSoftmaxKernelT() = default;

    // ============================================================================
    // Explicit Template Instantiations
    // ============================================================================

    template class CPUSoftmaxKernelT<ActivationPrecision::FP32>;
    template class CPUSoftmaxKernelT<ActivationPrecision::BF16>;
    template class CPUSoftmaxKernelT<ActivationPrecision::FP16>;
    template class CPUSoftmaxKernelT<ActivationPrecision::Q8_1>;

} // namespace llaminar2
