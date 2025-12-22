/**
 * @file GemmOp.h
 * @brief Typed self-validating GEMM (matrix multiplication) operation
 *
 * Template-based operation supporting all activation precisions (FP32, BF16, FP16, Q8_1).
 * Uses compile-time dispatch via ActivationPrecision enum for zero-overhead precision handling.
 *
 * Encapsulates the full weight GEMM workflow:
 * 1. Validate activation/weight/output tensors
 * 2. Get/create GEMM kernel from weight tensor (cached via KernelFactory)
 * 3. Execute kernel with precision-appropriate paths
 * 4. Capture snapshot (if enabled)
 *
 * Supports both:
 * - Weight GEMM: C = A @ W^T (activations × weight tensor)
 * - Activation GEMM: C = A @ B^T (activations × activations)
 *
 * Usage:
 * @code
 * // Create typed op at initialization (based on config.activation_precision)
 * auto gemm = createGemmOp(ActivationPrecision::Q8_1);
 *
 * // Weight projection (polymorphic call, type-specific implementation)
 * TRY_OP(gemm->execute(hidden, layer.wq.get(), Q_buf, seq_len, n_heads * head_dim, d_model,
 *                      nullptr, device));
 *
 * @endcode
 *
 * @author David Sanftenberg
 */

#pragma once

#include "Op.h"
#include "../RuntimeConfig.h"
#include "../../kernels/KernelFactory.h"
#include "../../tensors/Tensors.h"
#include <memory>

namespace llaminar2
{

    // =========================================================================
    // Precision-to-Tensor Type Mapping (shared across ops)
    // =========================================================================

    namespace detail
    {
        template <ActivationPrecision P>
        struct PrecisionTensor;

        template <>
        struct PrecisionTensor<ActivationPrecision::FP32>
        {
            using Type = FP32Tensor;
            using ElementType = float;
        };

        template <>
        struct PrecisionTensor<ActivationPrecision::BF16>
        {
            using Type = BF16Tensor;
            using ElementType = uint16_t;
        };

        template <>
        struct PrecisionTensor<ActivationPrecision::FP16>
        {
            using Type = FP16Tensor;
            using ElementType = uint16_t;
        };

        template <>
        struct PrecisionTensor<ActivationPrecision::Q8_1>
        {
            using Type = Q8_1Tensor;
            using ElementType = Q8_1Block;
        };
    } // namespace detail

    // =========================================================================
    // IGemmOp Interface
    // =========================================================================

    /**
     * @brief Interface for typed GEMM operations
     *
     * Provides polymorphic access to precision-specific GEMM implementations.
     * Use createGemmOp() factory to create instances at runtime.
     */
    class IGemmOp
    {
    public:
        virtual ~IGemmOp() = default;

        /**
         * @brief Get the activation precision this op was created for
         */
        virtual ActivationPrecision precision() const = 0;

        /**
         * @brief Execute weight projection: C = A @ W^T
         *
         * @param A Input activations tensor [m, k]
         * @param W Weight tensor [n, k] (transposed storage)
         * @param C Output tensor [m, n]
         * @param m Number of rows (sequence length)
         * @param n Number of output features
         * @param k Number of input features
         * @param mpi_ctx MPI context (nullptr for single-node)
         * @param device_idx Device index (-1 for CPU)
         * @param alpha Scale factor for A@W (default: 1.0f)
         * @param beta Scale factor for existing C (default: 0.0f)
         *
         * @return true on success, false on validation or execution failure
         */
        virtual bool execute(
            const TensorBase *A,
            TensorBase *W,
            TensorBase *C,
            int m, int n, int k,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            float alpha = 1.0f,
            float beta = 0.0f) = 0;

        /**
         * @brief Execute activation-activation matmul: C = A @ B^T (or A @ B)
         *
         * @param A Left activation tensor [m, k]
         * @param B Right activation tensor [n, k] if transpose_B, [k, n] otherwise
         * @param C Output tensor [m, n]
         * @param m Number of rows in A
         * @param n Number of rows in B (if transpose_B) or columns in B
         * @param k Inner dimension
         * @param transpose_B Whether to transpose B (true for Q@K^T)
         * @param alpha Scale factor (e.g., 1/sqrt(d_k) for attention)
         * @param beta Scale factor for existing C
         * @param mpi_ctx MPI context (nullptr for single-node)
         * @param device_idx Device index (-1 for CPU)
         *
         * @return true on success, false on failure
         */
        virtual bool activations(
            TensorBase *A,
            TensorBase *B,
            TensorBase *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f,
            float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;

        /**
         * @brief Execute weight projection with raw float pointers
         *
         * Alternative for when activation data is not in a TensorBase.
         * Note: Only supports FP32 data.
         */
        virtual bool execute_raw(
            const float *A_data,
            TensorBase *W,
            float *C_data,
            int m, int n, int k,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            float alpha = 1.0f,
            float beta = 0.0f) = 0;
    };

    // =========================================================================
    // GemmOpTyped<Precision> Template Implementation
    // =========================================================================

    /**
     * @brief Typed GEMM operation with compile-time precision dispatch
     *
     * @tparam Precision Activation precision (FP32, BF16, FP16, Q8_1)
     */
    template <ActivationPrecision Precision>
    class GemmOpTyped : public IGemmOp, public OpBase
    {
    public:
        using TensorT = typename detail::PrecisionTensor<Precision>::Type;
        using ElementType = typename detail::PrecisionTensor<Precision>::ElementType;

        const char *name() const override { return "GemmOpTyped"; }
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
            const TensorBase *A,
            TensorBase *W,
            TensorBase *C,
            int m, int n, int k,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            float alpha = 1.0f,
            float beta = 0.0f) override
        {
            // 1. Validate inputs
            if (!validateTensor(A, "activation (A)"))
                return false;
            if (!validateTensor(W, "weight (W)"))
                return false;
            if (!validateTensor(C, "output (C)"))
                return false;
            if (!validateDimensions(m, k, "input A"))
                return false;
            if (!validateDimensions(m, n, "output C"))
                return false;

            // 2. Type validation: ensure activation tensor matches expected precision
            //    Output tensor can be either the expected type OR FP32.
            //    FP32 output is allowed for:
            //    - LM head logits (always FP32 for sampling accuracy)
            //    - Mixed-precision paths where kernel dequantizes on output
            constexpr TensorType expected = expected_tensor_type();
            if (A->native_type() != expected)
            {
                logError(("activation tensor type mismatch: expected " +
                          std::to_string(static_cast<int>(expected)) + ", got " +
                          std::to_string(static_cast<int>(A->native_type())))
                             .c_str());
                return false;
            }
            // Allow output to be either the expected type or FP32
            // (kernel handles dequantization when output is FP32)
            if (C->native_type() != expected && C->native_type() != TensorType::FP32)
            {
                logError(("output tensor type mismatch: expected " +
                          std::to_string(static_cast<int>(expected)) + " or FP32, got " +
                          std::to_string(static_cast<int>(C->native_type())))
                             .c_str());
                return false;
            }

            // 3. Get cached kernel from KernelFactory
            auto *gemm_kernel = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(W);
            if (!gemm_kernel)
            {
                logError("failed to get GEMM kernel from weight tensor");
                return false;
            }

            // 4. Delegate to kernel's multiply_tensor with explicit dimensions
            //    CRITICAL: Use m,n,k parameters, NOT tensor shapes!
            //    Tensor shapes may be larger (pre-allocated buffers) than actual data.
            //    The kernel inspects A->native_type() and C->native_type() to choose optimal path:
            //    - Q8_1 → Q8_1: Zero-copy quantized path
            //    - Q8_1 → FP32: Direct Q8_1 input, FP32 output
            //    - FP32 → Q8_1: Quantize output on-the-fly
            //    - FP32 → FP32: Standard path
            if (!gemm_kernel->multiply_tensor(A, C, m, n, k, true, alpha, beta, mpi_ctx, device_idx))
            {
                logError("GEMM kernel multiply_tensor failed");
                return false;
            }

            return true;
        }

        bool activations(
            TensorBase *A,
            TensorBase *B,
            TensorBase *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f,
            float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            // 1. Validate inputs
            if (!validateTensor(A, "activation A"))
                return false;
            if (!validateTensor(B, "activation B"))
                return false;
            if (!validateTensor(C, "output C"))
                return false;
            if (!validateDimensions(m, k, "input A"))
                return false;

            // 2. Type validation: ensure A and B match expected precision
            //    Note: C is typically FP32 for attention scores regardless of activation precision
            constexpr TensorType expected = expected_tensor_type();
            if (A->native_type() != expected)
            {
                logError(("activation A tensor type mismatch: expected " +
                          std::to_string(static_cast<int>(expected)) + ", got " +
                          std::to_string(static_cast<int>(A->native_type())))
                             .c_str());
                return false;
            }
            if (B->native_type() != expected)
            {
                logError(("activation B tensor type mismatch: expected " +
                          std::to_string(static_cast<int>(expected)) + ", got " +
                          std::to_string(static_cast<int>(B->native_type())))
                             .c_str());
                return false;
            }

            // 3. Create kernel from A
            auto gemm_kernel = A->createGemm();
            if (!gemm_kernel)
            {
                logError("failed to create GEMM kernel for activation matmul");
                return false;
            }

            // 4. Delegate to kernel's multiply_activations_tensor - handles all type dispatch
            //    The kernel inspects A/B/C native_type() to choose optimal path:
            //    - FP32 × FP32: Direct FP32 matmul
            //    - BF16 × BF16: OneDNN bf16bf16f32
            //    - Q8_1 × Q8_1: Dequant to FP32 (attention scores always FP32)
            if (!gemm_kernel->multiply_activations_tensor(A, B, C, transpose_B, alpha, beta, mpi_ctx, device_idx))
            {
                logError("activation GEMM kernel multiply_activations_tensor failed");
                return false;
            }

            return true;
        }

        bool execute_raw(
            const float *A_data,
            TensorBase *W,
            float *C_data,
            int m, int n, int k,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            float alpha = 1.0f,
            float beta = 0.0f) override
        {
            // 1. Validate inputs
            if (!validatePointer(A_data, "activation data (A)"))
                return false;
            if (!validateTensor(W, "weight (W)"))
                return false;
            if (!validatePointer(C_data, "output data (C)"))
                return false;

            // 2. Get cached kernel from KernelFactory
            auto *gemm_kernel = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(W);
            if (!gemm_kernel)
            {
                logError("failed to get GEMM kernel from weight tensor");
                return false;
            }

            // 3. Execute (raw always uses FP32)
            if (!gemm_kernel->multiply_activations(
                    A_data,
                    nullptr,
                    C_data,
                    m, n, k,
                    true,
                    alpha, beta,
                    mpi_ctx,
                    device_idx))
            {
                logError("GEMM kernel execution failed");
                return false;
            }

            return true;
        }
    };

    // =========================================================================
    // Factory Function
    // =========================================================================

    /**
     * @brief Create typed GEMM op for given precision
     *
     * @param precision Activation precision
     * @return Unique pointer to precision-specific GEMM op
     */
    inline std::unique_ptr<IGemmOp> createGemmOp(ActivationPrecision precision)
    {
        switch (precision)
        {
        case ActivationPrecision::Q8_1:
            return std::make_unique<GemmOpTyped<ActivationPrecision::Q8_1>>();
        case ActivationPrecision::BF16:
            return std::make_unique<GemmOpTyped<ActivationPrecision::BF16>>();
        case ActivationPrecision::FP16:
            return std::make_unique<GemmOpTyped<ActivationPrecision::FP16>>();
        case ActivationPrecision::FP32:
        default:
            return std::make_unique<GemmOpTyped<ActivationPrecision::FP32>>();
        }
    }

} // namespace llaminar2
