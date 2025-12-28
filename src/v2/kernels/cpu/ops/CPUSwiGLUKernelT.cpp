/**
 * @file CPUSwiGLUKernelT.cpp
 * @brief Implementation of typed SwiGLU kernel with ActivationPrecision support
 * @author David Sanftenberg
 *
 * This file implements the CPUSwiGLUKernelT template specializations for
 * FP32, BF16, FP16, and Q8_1 precision types.
 *
 * Each precision uses native primitives without unnecessary type conversions:
 * - FP32: Direct FP32 SwiGLU with AVX512/AVX2 vectorization
 * - BF16: Native BF16 SwiGLU (converts to FP32 for silu, back to BF16)
 * - FP16: Native FP16 SwiGLU (converts to FP32 for silu, back to FP16)
 * - Q8_1: Integer-aware SwiGLU (dequant, compute, requant per block)
 */

#include "CPUSwiGLUKernelT.h"
#include "../primitives/SwiGLUPrimitives.h"
#include "../../../tensors/Tensors.h" // For TensorBase
#include "../../../utils/Logger.h"

namespace llaminar2
{

    // ============================================================================
    // FP32 Specialization Implementation
    // ============================================================================

    bool CPUSwiGLUKernelT<ActivationPrecision::FP32>::apply_typed(
        const float *gate,
        const float *up,
        float *output,
        int size,
        int device_idx)
    {
        (void)device_idx; // Unused for CPU kernel

        if (!gate || !up || !output)
        {
            LOG_ERROR("CPUSwiGLUKernelT<FP32>: Null data pointer");
            return false;
        }

        if (size <= 0)
        {
            LOG_DEBUG("CPUSwiGLUKernelT<FP32>: Empty input (size=" << size << ")");
            return true; // Nothing to do
        }

        // Use vectorized SwiGLU primitive (AVX512/AVX2 with OpenMP)
        primitives::compute_swiglu(gate, up, output, size);

        return true;
    }

    // ============================================================================
    // BF16 Specialization Implementation
    // ============================================================================

    bool CPUSwiGLUKernelT<ActivationPrecision::BF16>::apply_typed(
        const uint16_t *gate,
        const uint16_t *up,
        uint16_t *output,
        int size,
        int device_idx)
    {
        (void)device_idx; // Unused for CPU kernel

        if (!gate || !up || !output)
        {
            LOG_ERROR("CPUSwiGLUKernelT<BF16>: Null data pointer");
            return false;
        }

        if (size <= 0)
        {
            LOG_DEBUG("CPUSwiGLUKernelT<BF16>: Empty input (size=" << size << ")");
            return true; // Nothing to do
        }

        // Use native BF16 SwiGLU primitive
        // Internally converts to FP32 for silu computation, converts back to BF16
        primitives::compute_swiglu_bf16(gate, up, output, size);

        return true;
    }

    // ============================================================================
    // FP16 Specialization Implementation
    // ============================================================================

    bool CPUSwiGLUKernelT<ActivationPrecision::FP16>::apply_typed(
        const uint16_t *gate,
        const uint16_t *up,
        uint16_t *output,
        int size,
        int device_idx)
    {
        (void)device_idx; // Unused for CPU kernel

        if (!gate || !up || !output)
        {
            LOG_ERROR("CPUSwiGLUKernelT<FP16>: Null data pointer");
            return false;
        }

        if (size <= 0)
        {
            LOG_DEBUG("CPUSwiGLUKernelT<FP16>: Empty input (size=" << size << ")");
            return true; // Nothing to do
        }

        // Use native FP16 SwiGLU primitive
        // Internally converts to FP32 for silu computation, converts back to FP16
        primitives::compute_swiglu_fp16(gate, up, output, size);

        return true;
    }

    // ============================================================================
    // Q8_1 Specialization Implementation (Integer-Aware)
    // ============================================================================

    bool CPUSwiGLUKernelT<ActivationPrecision::Q8_1>::apply_typed(
        const Q8_1Block *gate,
        const Q8_1Block *up,
        Q8_1Block *output,
        int size,
        int device_idx)
    {
        (void)device_idx; // Unused for CPU kernel

        if (!gate || !up || !output)
        {
            LOG_ERROR("CPUSwiGLUKernelT<Q8_1>: Null data pointer");
            return false;
        }

        if (size <= 0)
        {
            LOG_DEBUG("CPUSwiGLUKernelT<Q8_1>: Empty input (size=" << size << ")");
            return true; // Nothing to do
        }

        // Use Q8_1 SwiGLU primitive
        // This handles:
        //   - Block-wise dequantization
        //   - SwiGLU computation: silu(gate) * up
        //   - Requantization with new scale factor
        primitives::compute_swiglu_q8_1(gate, up, output, size);

        return true;
    }

    // ============================================================================
    // Destructor Definitions (required for vtable emission)
    // ============================================================================

    CPUSwiGLUKernelT<ActivationPrecision::FP32>::~CPUSwiGLUKernelT() = default;
    CPUSwiGLUKernelT<ActivationPrecision::BF16>::~CPUSwiGLUKernelT() = default;
    CPUSwiGLUKernelT<ActivationPrecision::FP16>::~CPUSwiGLUKernelT() = default;
    CPUSwiGLUKernelT<ActivationPrecision::Q8_1>::~CPUSwiGLUKernelT() = default;

    // ============================================================================
    // ITensorSwiGLU Interface Implementations - FP32
    // ============================================================================

    bool CPUSwiGLUKernelT<ActivationPrecision::FP32>::apply(
        const float *gate, const float *up, float *output,
        int rows, int cols,
        bool add_residual,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)add_residual; // Not used in this kernel
        (void)mpi_ctx;      // Not used in this kernel
        const int size = rows * cols;
        return apply_typed(gate, up, output, size, device_idx);
    }

    bool CPUSwiGLUKernelT<ActivationPrecision::FP32>::apply_tensor(
        const TensorBase *gate,
        const TensorBase *up,
        TensorBase *output,
        int rows, int cols,
        bool add_residual,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if (!gate || !up || !output)
        {
            LOG_ERROR("CPUSwiGLUKernelT<FP32>::apply_tensor: null tensor");
            return false;
        }

        // Validate all tensors are FP32
        if (gate->native_type() != TensorType::FP32 ||
            up->native_type() != TensorType::FP32 ||
            output->native_type() != TensorType::FP32)
        {
            LOG_ERROR("CPUSwiGLUKernelT<FP32>::apply_tensor: type mismatch, expected FP32");
            return false;
        }

        // FP32Tensor::data() returns const float*, FP32Tensor::mutable_data() returns float*
        return apply(
            gate->data(),
            up->data(),
            output->mutable_data(),
            rows, cols, add_residual, mpi_ctx, device_idx);
    }

    // ============================================================================
    // ITensorSwiGLU Interface Implementations - BF16
    // ============================================================================

    bool CPUSwiGLUKernelT<ActivationPrecision::BF16>::apply(
        const float *gate, const float *up, float *output,
        int rows, int cols,
        bool add_residual,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)gate;
        (void)up;
        (void)output;
        (void)rows;
        (void)cols;
        (void)add_residual;
        (void)mpi_ctx;
        (void)device_idx;
        LOG_ERROR("CPUSwiGLUKernelT<BF16>::apply(FP32): use apply_bf16 for BF16 data");
        return false;
    }

    bool CPUSwiGLUKernelT<ActivationPrecision::BF16>::apply_bf16(
        const uint16_t *gate, const uint16_t *up, uint16_t *output,
        int rows, int cols,
        bool add_residual,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)add_residual; // Not used in this kernel
        (void)mpi_ctx;      // Not used in this kernel
        const int size = rows * cols;
        return apply_typed(gate, up, output, size, device_idx);
    }

    bool CPUSwiGLUKernelT<ActivationPrecision::BF16>::apply_tensor(
        const TensorBase *gate,
        const TensorBase *up,
        TensorBase *output,
        int rows, int cols,
        bool add_residual,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if (!gate || !up || !output)
        {
            LOG_ERROR("CPUSwiGLUKernelT<BF16>::apply_tensor: null tensor");
            return false;
        }

        // Validate all tensors are BF16
        if (gate->native_type() != TensorType::BF16 ||
            up->native_type() != TensorType::BF16 ||
            output->native_type() != TensorType::BF16)
        {
            LOG_ERROR("CPUSwiGLUKernelT<BF16>::apply_tensor: type mismatch, expected BF16");
            return false;
        }

        // Cast to BF16Tensor to access typed_data() / mutable_typed_data()
        auto *gate_bf16 = dynamic_cast<const BF16Tensor *>(gate);
        auto *up_bf16 = dynamic_cast<const BF16Tensor *>(up);
        auto *output_bf16 = dynamic_cast<BF16Tensor *>(output);

        if (!gate_bf16 || !up_bf16 || !output_bf16)
        {
            LOG_ERROR("CPUSwiGLUKernelT<BF16>::apply_tensor: failed to cast tensors to BF16Tensor");
            return false;
        }

        return apply_bf16(
            gate_bf16->typed_data(),
            up_bf16->typed_data(),
            output_bf16->mutable_typed_data(),
            rows, cols, add_residual, mpi_ctx, device_idx);
    }

    // ============================================================================
    // ITensorSwiGLU Interface Implementations - FP16
    // ============================================================================

    bool CPUSwiGLUKernelT<ActivationPrecision::FP16>::apply(
        const float *gate, const float *up, float *output,
        int rows, int cols,
        bool add_residual,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)gate;
        (void)up;
        (void)output;
        (void)rows;
        (void)cols;
        (void)add_residual;
        (void)mpi_ctx;
        (void)device_idx;
        LOG_ERROR("CPUSwiGLUKernelT<FP16>::apply(FP32): use apply_fp16 for FP16 data");
        return false;
    }

    bool CPUSwiGLUKernelT<ActivationPrecision::FP16>::apply_fp16(
        const uint16_t *gate, const uint16_t *up, uint16_t *output,
        int rows, int cols,
        bool add_residual,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)add_residual; // Not used in this kernel
        (void)mpi_ctx;      // Not used in this kernel
        const int size = rows * cols;
        return apply_typed(gate, up, output, size, device_idx);
    }

    bool CPUSwiGLUKernelT<ActivationPrecision::FP16>::apply_tensor(
        const TensorBase *gate,
        const TensorBase *up,
        TensorBase *output,
        int rows, int cols,
        bool add_residual,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if (!gate || !up || !output)
        {
            LOG_ERROR("CPUSwiGLUKernelT<FP16>::apply_tensor: null tensor");
            return false;
        }

        // Validate all tensors are FP16
        if (gate->native_type() != TensorType::FP16 ||
            up->native_type() != TensorType::FP16 ||
            output->native_type() != TensorType::FP16)
        {
            LOG_ERROR("CPUSwiGLUKernelT<FP16>::apply_tensor: type mismatch, expected FP16");
            return false;
        }

        // Cast to FP16Tensor to access typed_data() / mutable_typed_data()
        auto *gate_fp16 = dynamic_cast<const FP16Tensor *>(gate);
        auto *up_fp16 = dynamic_cast<const FP16Tensor *>(up);
        auto *output_fp16 = dynamic_cast<FP16Tensor *>(output);

        if (!gate_fp16 || !up_fp16 || !output_fp16)
        {
            LOG_ERROR("CPUSwiGLUKernelT<FP16>::apply_tensor: failed to cast tensors to FP16Tensor");
            return false;
        }

        return apply_fp16(
            gate_fp16->typed_data(),
            up_fp16->typed_data(),
            output_fp16->mutable_typed_data(),
            rows, cols, add_residual, mpi_ctx, device_idx);
    }

    // ============================================================================
    // ITensorSwiGLU Interface Implementations - Q8_1
    // ============================================================================

    bool CPUSwiGLUKernelT<ActivationPrecision::Q8_1>::apply(
        const float *gate, const float *up, float *output,
        int rows, int cols,
        bool add_residual,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)gate;
        (void)up;
        (void)output;
        (void)rows;
        (void)cols;
        (void)add_residual;
        (void)mpi_ctx;
        (void)device_idx;
        LOG_ERROR("CPUSwiGLUKernelT<Q8_1>::apply(FP32): use apply_q8_1 for Q8_1 data");
        return false;
    }

    bool CPUSwiGLUKernelT<ActivationPrecision::Q8_1>::apply_q8_1(
        const void *gate, const void *up, void *output,
        int rows, int cols,
        bool add_residual,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)add_residual; // Not used in this kernel
        (void)mpi_ctx;      // Not used in this kernel
        const int size = rows * cols;
        return apply_typed(
            static_cast<const Q8_1Block *>(gate),
            static_cast<const Q8_1Block *>(up),
            static_cast<Q8_1Block *>(output),
            size, device_idx);
    }

    bool CPUSwiGLUKernelT<ActivationPrecision::Q8_1>::apply_tensor(
        const TensorBase *gate,
        const TensorBase *up,
        TensorBase *output,
        int rows, int cols,
        bool add_residual,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if (!gate || !up || !output)
        {
            LOG_ERROR("CPUSwiGLUKernelT<Q8_1>::apply_tensor: null tensor");
            return false;
        }

        // Validate all tensors are Q8_1
        if (gate->native_type() != TensorType::Q8_1 ||
            up->native_type() != TensorType::Q8_1 ||
            output->native_type() != TensorType::Q8_1)
        {
            LOG_ERROR("CPUSwiGLUKernelT<Q8_1>::apply_tensor: type mismatch, expected Q8_1");
            return false;
        }

        // Cast to Q8_1Tensor to access typed_data() / mutable_typed_data()
        auto *gate_q8 = dynamic_cast<const Q8_1Tensor *>(gate);
        auto *up_q8 = dynamic_cast<const Q8_1Tensor *>(up);
        auto *output_q8 = dynamic_cast<Q8_1Tensor *>(output);

        if (!gate_q8 || !up_q8 || !output_q8)
        {
            LOG_ERROR("CPUSwiGLUKernelT<Q8_1>::apply_tensor: failed to cast tensors to Q8_1Tensor");
            return false;
        }

        return apply_q8_1(
            gate_q8->typed_data(),
            up_q8->typed_data(),
            output_q8->mutable_typed_data(),
            rows, cols, add_residual, mpi_ctx, device_idx);
    }

} // namespace llaminar2
