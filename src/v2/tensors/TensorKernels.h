/**
 * @file TensorKernels.h
 * @brief Kernel interfaces for tensor operations
 *
 * All kernels accept MPIContext (distributed coordination) and device_idx (execution device).
 * Kernels are created by tensors via createGemm(), createRoPE(), etc.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../utils/MPIContext.h"
#include <memory>

namespace llaminar2
{
    // Forward declaration for tensor-based GEMM interface
    class TensorBase;

    // =============================================================================
    // Fused Operation Configuration
    // =============================================================================

    /**
     * @brief Fused operation types that can be applied after GEMM
     *
     * These operations are fused into the GEMM kernel to avoid extra memory passes.
     */
    enum class GemmFusedOpType : uint8_t
    {
        None = 0, ///< No post-GEMM fusion
        SwiGLU,   ///< SwiGLU activation: output *= swish(gate) = output * gate * sigmoid(gate)
        Softmax,  ///< Row-wise softmax: output = softmax(output + mask)
        ReLU,     ///< ReLU activation: output = max(0, output)
        GeLU,     ///< GeLU activation: output = output * 0.5 * (1 + erf(output / sqrt(2)))
        SiLU      ///< SiLU/Swish activation: output = output * sigmoid(output)
    };

    /**
     * @brief Configuration for fused operations after GEMM
     *
     * This struct consolidates all fused operation parameters into a single
     * extensible configuration object. Use the static factory methods for
     * convenient construction.
     *
     * Design rationale:
     * - Avoids interface bloat from separate methods for each fused op
     * - Extensible: add new fused ops without breaking API
     * - Zero-cost when unused (default-constructed = no fusion)
     *
     * Usage:
     * @code
     * // No fusion (default)
     * kernel->multiply_with_precomputed_q8_1(..., GemmFusedOps::none());
     *
     * // SwiGLU fusion
     * kernel->multiply_with_precomputed_q8_1(..., GemmFusedOps::swiglu(gate_output));
     *
     * // Softmax fusion (for attention)
     * kernel->multiply_with_precomputed_q8_1(..., GemmFusedOps::softmax(1.0f / sqrt(d_k), mask));
     * @endcode
     */
    struct GemmFusedOps
    {
        GemmFusedOpType op_type = GemmFusedOpType::None;

        // --- SwiGLU parameters ---
        const float *gate_input = nullptr; ///< Gate tensor for SwiGLU [m, n] (output *= swish(gate))

        // --- Softmax parameters ---
        float softmax_scale = 1.0f;          ///< Pre-softmax scale (typically 1/sqrt(d_k))
        const float *softmax_mask = nullptr; ///< Additive mask before softmax [m, n] or nullptr
        bool is_causal = false;              ///< Apply causal (lower-triangular) mask
        int softmax_axis = 1;                ///< Axis for softmax reduction (typically 1 = row-wise)

        // --- Online softmax outputs (for flash attention) ---
        float *online_max = nullptr; ///< Output: per-row max values [m]
        float *online_sum = nullptr; ///< Output: per-row sum of exp values [m]

        // =========================================================================
        // Factory methods for convenient construction
        // =========================================================================

        /**
         * @brief No fused operation (default)
         */
        static GemmFusedOps none()
        {
            return GemmFusedOps{};
        }

        /**
         * @brief SwiGLU fusion: output = output * swish(gate) = output * gate * sigmoid(gate)
         *
         * Used in FFN: up_output = up_proj * swish(gate_proj)
         *
         * @param gate Gate tensor [m, n] (must match output dimensions)
         */
        static GemmFusedOps swiglu(const float *gate)
        {
            GemmFusedOps ops;
            ops.op_type = GemmFusedOpType::SwiGLU;
            ops.gate_input = gate;
            return ops;
        }

        /**
         * @brief Softmax fusion: output = softmax(output * scale + mask)
         *
         * Used in attention: scores = softmax(Q @ K^T / sqrt(d_k) + mask)
         *
         * @param scale Pre-softmax scale (typically 1/sqrt(d_k))
         * @param mask Optional additive mask [m, n] (nullptr = no mask)
         * @param causal Apply causal (lower-triangular) mask
         */
        static GemmFusedOps softmax(float scale = 1.0f, const float *mask = nullptr, bool causal = false)
        {
            GemmFusedOps ops;
            ops.op_type = GemmFusedOpType::Softmax;
            ops.softmax_scale = scale;
            ops.softmax_mask = mask;
            ops.is_causal = causal;
            return ops;
        }

        /**
         * @brief Online softmax for flash attention (returns max/sum for rescaling)
         *
         * @param scale Pre-softmax scale
         * @param out_max Output buffer for per-row max [m]
         * @param out_sum Output buffer for per-row exp sum [m]
         * @param mask Optional additive mask
         * @param causal Apply causal mask
         */
        static GemmFusedOps online_softmax(float scale, float *out_max, float *out_sum,
                                           const float *mask = nullptr, bool causal = false)
        {
            GemmFusedOps ops;
            ops.op_type = GemmFusedOpType::Softmax;
            ops.softmax_scale = scale;
            ops.softmax_mask = mask;
            ops.is_causal = causal;
            ops.online_max = out_max;
            ops.online_sum = out_sum;
            return ops;
        }

        /**
         * @brief Check if any fused operation is enabled
         */
        bool has_fusion() const { return op_type != GemmFusedOpType::None; }

        /**
         * @brief Check if specific operation type is enabled
         */
        bool is_swiglu() const { return op_type == GemmFusedOpType::SwiGLU; }
        bool is_softmax() const { return op_type == GemmFusedOpType::Softmax; }
    };

    // =============================================================================
    // Activation Format Enumeration
    // =============================================================================

    /**
     * @brief Activation precision format enumeration
     *
     * Used by ITensorGemm to dispatch to appropriate GEMM implementation
     * based on activation and weight precision.
     *
     * Format semantics:
     * - FP32: 32-bit floating point (IEEE 754 binary32)
     * - BF16: Brain Float 16 (truncated FP32, 8-bit exponent)
     * - FP16: Half precision (IEEE 754 binary16)
     * - INT8: 8-bit integer (quantized activation, needs INT32 accumulator)
     * - Q8_0: Quantized 8-bit with per-block scales (weight format, not activation)
     * - Q8_1: Quantized 8-bit with per-block scales + zero points (activation format)
     */
    enum class ActivationFormat
    {
        FP32, ///< 32-bit float (native CPU, universal fallback)
        BF16, ///< Brain Float 16 (AVX512_BF16, OneDNN native)
        FP16, ///< Half precision (FP16 instructions, OneDNN native)
        INT8, ///< Quantized 8-bit integer (AVX512-VNNI, INT32 accumulator)
        Q8_0, ///< Quantized 8-bit weight format (block-scaled)
        Q8_1  ///< Quantized 8-bit activation format (block-scaled with zero-point)
    };

    /**
     * @brief Block decoder strategy interface for quantized tensors (FP32 decode path)
     *
     * Provides format-specific dequantization for block-quantized formats.
     * Implementations MUST be header-only with always_inline to ensure zero overhead
     * when called from GEMM hot paths.
     *
     * Supported formats: IQ4_NL, Q6_K, Q8_0, etc.
     *
     * NOTE: This interface decodes to FP32. For integer GEMM kernels that want
     * raw quantized blocks, use IQuantizedTileAccessor instead.
     */
    class ITensorGemmTileDataProvider
    {
    public:
        virtual ~ITensorGemmTileDataProvider() = default;

        /**
         * @brief Decode one quantized block to FP32
         *
         * CRITICAL: This method is called in GEMM hot path (thousands of times per matmul).
         * Must be marked always_inline and implemented in header for zero overhead.
         *
         * @param row_idx Row index in weight tensor
         * @param k_block_offset Block offset along K dimension (units of BLOCK_SIZE)
         * @param output Output buffer (must have space for BLOCK_SIZE floats, typically 32)
         */
        __attribute__((always_inline)) virtual void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const = 0;

        /**
         * @brief Get raw pointer to quantized block (for VNNI/int8 optimizations)
         *
         * @param row_idx Row index in weight tensor
         * @param k_block_offset Block offset along K dimension
         * @return Const pointer to raw quantized block structure
         */
        virtual const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const = 0;

        /**
         * @brief Get tensor dimensions
         */
        virtual size_t decoder_rows() const = 0;
        virtual size_t decoder_cols() const = 0;
        virtual size_t block_size() const = 0;
    };

    /**
     * @brief Raw quantized weight block accessor (integer GEMM path)
     *
     * Unlike ITensorGemmTileDataProvider (which decodes to FP32), this interface
     * exposes raw quantized blocks + metadata for integer GEMM kernels that want
     * to defer dequantization (e.g., AVX512-VNNI INT8×IQ4_NL GEMM).
     *
     * Rationale:
     * - FP32 GEMM: Decode weights to FP32 → multiply with FP32 activations
     * - INT8 GEMM: Keep weights as int8, decode on-the-fly in registers,
     *              apply scaling to final output (fused dequant)
     *
     * This avoids materializing FP32 weight buffers for integer kernels.
     */
    class IQuantizedTileAccessor
    {
    public:
        virtual ~IQuantizedTileAccessor() = default;

        /**
         * @brief Get raw pointer to quantized block (no decode)
         *
         * @param row_idx Row index in weight tensor
         * @param k_block_idx Block index along K dimension
         * @return Const pointer to raw quantized block structure
         *
         * Example: For IQ4_NL, returns IQ4_NLBlock* = {uint8_t qs[16], uint16_t d}
         */
        virtual const void *get_raw_block(size_t row_idx, size_t k_block_idx) const = 0;

        /**
         * @brief Get dequantization scale for block
         *
         * @param row_idx Row index
         * @param k_block_idx Block index
         * @return FP32 scale factor (converted from FP16 if needed)
         */
        virtual float get_block_scale(size_t row_idx, size_t k_block_idx) const = 0;

        /**
         * @brief Get tensor dimensions
         */
        virtual size_t rows() const = 0;
        virtual size_t cols() const = 0;
        virtual size_t block_size() const = 0;
    };

    /**
     * @brief Base kernel interface
     */
    class ITensorKernel
    {
    public:
        virtual ~ITensorKernel() = default;

        /**
         * @brief Check if kernel supports specific backend
         *
         * @param device_idx Device index (-1 = CPU, ≥0 = GPU)
         * @return true if kernel can execute on this device
         */
        virtual bool supports_device(int device_idx) const = 0;
    };

    /**
     * @brief Matrix multiplication kernel (GEMM)
     *
     * C = alpha * A @ B + beta * C
     *
     * Implementations:
     * - CPUGemmKernel: OpenBLAS/MKL
     * - CUDAGemmKernel: cuBLAS + fused dequant for IQ4_NL
     * - ROCmGemmKernel: hipBLAS + fused dequant
     */
    class ITensorGemm : public ITensorKernel
    {
    public:
        /**
         * @brief Matrix multiplication: C = alpha * A @ B + beta * C
         *
         * @param A Input activations [m, k] (FP32, host or device)
         * @param C Output matrix [m, n] (FP32, host or device)
         * @param m Number of rows in A and C
         * @param n Number of columns in B and C (before transpose)
         * @param k Number of columns in A and rows in B
         * @param transpose_B Whether B is stored transposed (typical for weights)
         * @param alpha Scale factor for A@B
         * @param beta Scale factor for existing C (for fused add)
         * @param mpi_ctx MPI context for distributed execution (nullptr = single node)
         * @param device_idx Device index for execution (-1 = host/CPU, ≥0 = GPU device)
         *
         * @return true on success, false on error
         *
         * @note A and C locations (host/device) must match device_idx:
         *       - device_idx = -1: A/C on host memory
         *       - device_idx ≥ 0: A/C on device memory (caller manages transfers)
         *
         * @note Kernel executes on device corresponding to device_idx.
         *       Weight tensor (B) is accessed via this->tensor which has its own device_idx.
         *       If weight is on different device than activation, kernel handles transfer.
         */
        virtual bool multiply(
            const float *A, float *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;

        /**
         * @brief Tensor-based matrix multiplication: C = alpha * A @ B + beta * C
         *
         * This interface accepts tensors directly, enabling:
         * - **Type introspection**: Kernel can detect activation format (FP32, Q8_1, BF16, etc.)
         * - **Zero-copy quantized path**: If A is Q8_1Tensor, skip float→Q8_1 conversion
         * - **Flexible output**: C can be any precision (FP32, BF16, etc.)
         *
         * **Dispatch logic** (typical implementation):
         * 1. Check A->native_type() to determine activation format
         * 2. If Q8_1: Use quantized blocks directly (via IQ8_1Decodable or block access)
         * 3. If FP32: Fall back to multiply(A->data(), C->mutable_data(), ...)
         * 4. If BF16: Convert or use native BF16 GEMM if available
         *
         * @param A Input activations tensor [m, k] (any format)
         * @param C Output tensor [m, n] (must be mutable, typically FP32)
         * @param transpose_B Whether B (packed weights) is transposed (typical: true)
         * @param alpha Scale factor for A@B
         * @param beta Scale factor for existing C (for fused add)
         * @param mpi_ctx MPI context for distributed execution
         * @param device_idx Device index for execution
         *
         * @return true on success, false if format combination not supported
         *
         * @note Default implementation returns false (not supported).
         *       Kernels that include Tensors.h can override with optimized paths.
         */
        virtual bool multiply_tensor(
            const TensorBase *A, TensorBase *C,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            // Default: not supported (TensorBase is only forward-declared here)
            // Subclasses that include Tensors.h can override with real implementation
            (void)A;
            (void)C;
            (void)transpose_B;
            (void)alpha;
            (void)beta;
            (void)mpi_ctx;
            (void)device_idx;
            return false;
        }

        /**
         * @brief Activation-activation matrix multiplication: C = alpha * A @ B^T + beta * C
         *
         * This variant supports both A and B as activation buffers (not weight tensors).
         * Useful for attention computation: Q @ K^T and scores @ V.
         *
         * @param A Left activation matrix [m, k] (FP32)
         * @param B Right activation matrix [n, k] if transpose_B=true, [k, n] if false (FP32)
         * @param C Output matrix [m, n] (FP32)
         * @param m Number of rows in A and C
         * @param n Number of rows in B (if transpose_B=true) or columns in B (if false)
         * @param k Number of columns in A and B (if transpose_B=true)
         * @param transpose_B Whether to transpose B (true for attention Q@K^T, false for scores@V)
         * @param alpha Scale factor for A@B
         * @param beta Scale factor for existing C (for fused add)
         * @param mpi_ctx MPI context for distributed execution (nullptr = single node)
         * @param device_idx Device index for execution (-1 = host/CPU, ≥0 = GPU device)
         *
         * @return true on success, false on error
         *
         * @note All buffers (A, B, C) must be on same device as device_idx:
         *       - device_idx = -1: All on host memory
         *       - device_idx ≥ 0: All on device memory
         */
        virtual bool multiply_activations(
            const float *A, const float *B, float *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;

        /**
         * @brief Strided activation-activation GEMM with custom leading dimensions
         *
         * Supports strided memory access for efficient multi-head attention without copying.
         * C = alpha * A @ B^T + beta * C (if transpose_B=true)
         * C = alpha * A @ B + beta * C (if transpose_B=false)
         *
         * @param A Left activation matrix with stride lda
         * @param B Right activation matrix with stride ldb
         * @param C Output matrix with stride ldc
         * @param m Number of rows in A and C
         * @param n Number of rows in B (transpose_B=true) or cols in B (transpose_B=false)
         * @param k Number of columns in A and B (transpose_B=true)
         * @param lda Leading dimension of A (stride between rows, typically ≥k)
         * @param ldb Leading dimension of B (stride between rows)
         * @param ldc Leading dimension of C (stride between rows, typically ≥n)
         * @param transpose_B Whether to transpose B
         * @param alpha Scale factor for A@B
         * @param beta Scale factor for existing C
         * @param mpi_ctx MPI context (nullptr = single node)
         * @param device_idx Device index (-1 = CPU, ≥0 = GPU)
         *
         * @return true on success, false on error
         *
         * @note Use case: Multi-head attention where Q/K/V heads are interleaved in memory.
         *       Instead of copying each head to contiguous buffer, use strides to access directly.
         *       Example: Q_all [num_heads, seq_len, head_dim] with lda = num_heads * head_dim
         */
        virtual bool multiply_activations_strided(
            const float *A, const float *B, float *C,
            int m, int n, int k,
            int lda, int ldb, int ldc,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;

        /**
         * @brief Virtual implementation for typed activation GEMM (type-erased)
         */
        virtual bool multiply_activations_typed_impl(
            const void *A, const void *B, float *C,
            int m, int n, int k,
            bool transpose_B,
            float alpha, float beta,
            const MPIContext *mpi_ctx,
            int device_idx,
            ActivationFormat format_A, ActivationFormat format_B)
        {
            (void)A;
            (void)B;
            (void)C;
            (void)m;
            (void)n;
            (void)k;
            (void)transpose_B;
            (void)alpha;
            (void)beta;
            (void)mpi_ctx;
            (void)device_idx;
            (void)format_A;
            (void)format_B;
            return false;
        }

        /**
         * @brief Virtual implementation for typed strided activation GEMM (type-erased)
         */
        virtual bool multiply_activations_strided_typed_impl(
            const void *A, const void *B, float *C,
            int m, int n, int k,
            int lda, int ldb, int ldc,
            bool transpose_B,
            float alpha, float beta,
            const MPIContext *mpi_ctx,
            int device_idx,
            ActivationFormat format_A, ActivationFormat format_B)
        {
            (void)A;
            (void)B;
            (void)C;
            (void)m;
            (void)n;
            (void)k;
            (void)lda;
            (void)ldb;
            (void)ldc;
            (void)transpose_B;
            (void)alpha;
            (void)beta;
            (void)mpi_ctx;
            (void)device_idx;
            (void)format_A;
            (void)format_B;
            return false;
        }

        /**
         * @brief Template-based typed activation-activation GEMM
         *
         * Supports native-precision GEMM without forced FP32 conversion.
         * C = alpha * A @ B^T + beta * C (typed precision)
         *
         * **Precision Combinations**:
         * - (FP32, FP32) → FP32 GEMM (OpenBLAS/OneDNN)
         * - (BF16, BF16) → BF16 GEMM (OneDNN bf16bf16f32 on AVX512_BF16)
         * - (FP16, FP16) → FP16 GEMM (OneDNN fp16fp16f32)
         * - (INT8, INT8) → INT8 GEMM (OneDNN int8int8s32 on AVX512-VNNI)
         * - Mixed precisions → Convert to common format (lower precision preferred)
         *
         * @tparam ActT Activation element type (float, uint16_t, int8_t)
         * @tparam WeightT Weight element type (same options as ActT)
         *
         * @param A Left activation matrix [m, k]
         * @param B Right activation matrix [n, k] if transpose_B=true
         * @param C Output matrix [m, n] (always float* for now)
         * @param m Number of rows in A and C
         * @param n Number of rows in B (transpose_B=true) or cols (false)
         * @param k Number of columns in A and B (transpose_B=true)
         * @param transpose_B Whether to transpose B
         * @param alpha Scale factor for A@B
         * @param beta Scale factor for existing C
         * @param mpi_ctx MPI context (nullptr = single node)
         * @param device_idx Device index (-1 = CPU, ≥0 = GPU)
         * @param format Activation format hint (e.g. BF16 vs FP16 for uint16_t)
         *
         * @return true on success, false on error
         *
         * @note Default implementation returns false (not implemented).
         *       Kernels override to support specific precision combinations.
         */
        template <typename ActT, typename WeightT>
        bool multiply_activations_typed(
            const ActT *A, const WeightT *B, float *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            ActivationFormat format = ActivationFormat::FP32)
        {
            ActivationFormat fmt_A = (std::is_same_v<ActT, float>) ? ActivationFormat::FP32 : format;
            ActivationFormat fmt_B = (std::is_same_v<WeightT, float>) ? ActivationFormat::FP32 : format;
            return multiply_activations_typed_impl(
                static_cast<const void *>(A), static_cast<const void *>(B), C,
                m, n, k, transpose_B, alpha, beta, mpi_ctx, device_idx, fmt_A, fmt_B);
        }

        /**
         * @brief Template-based typed strided activation-activation GEMM
         *
         * Supports strided memory access with typed precision.
         * C = alpha * A @ B^T + beta * C (typed precision, strided)
         *
         * @tparam ActT Activation element type (float, uint16_t, int8_t)
         * @tparam WeightT Weight element type (same options as ActT)
         *
         * @param A Left activation matrix with stride lda
         * @param B Right activation matrix with stride ldb
         * @param C Output matrix with stride ldc (always float* for now)
         * @param m Number of rows in A and C
         * @param n Number of rows in B (transpose_B=true) or cols (false)
         * @param k Number of columns in A and B (transpose_B=true)
         * @param lda Leading dimension of A (stride between rows)
         * @param ldb Leading dimension of B (stride between rows)
         * @param ldc Leading dimension of C (stride between rows)
         * @param transpose_B Whether to transpose B
         * @param alpha Scale factor for A@B
         * @param beta Scale factor for existing C
         * @param mpi_ctx MPI context (nullptr = single node)
         * @param device_idx Device index (-1 = CPU, ≥0 = GPU)
         * @param format Activation format hint (e.g. BF16 vs FP16 for uint16_t)
         *
         * @return true on success, false on error
         *
         * @note Default implementation returns false (not implemented).
         *       Kernels override to support specific precision combinations.
         */
        template <typename ActT, typename WeightT>
        bool multiply_activations_strided_typed(
            const ActT *A, const WeightT *B, float *C,
            int m, int n, int k,
            int lda, int ldb, int ldc,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            ActivationFormat format = ActivationFormat::FP32)
        {
            ActivationFormat fmt_A = (std::is_same_v<ActT, float>) ? ActivationFormat::FP32 : format;
            ActivationFormat fmt_B = (std::is_same_v<WeightT, float>) ? ActivationFormat::FP32 : format;
            return multiply_activations_strided_typed_impl(
                static_cast<const void *>(A), static_cast<const void *>(B), C,
                m, n, k, lda, ldb, ldc, transpose_B, alpha, beta, mpi_ctx, device_idx, fmt_A, fmt_B);
        }

        /**
         * @brief Matrix multiplication with fused Softmax (for attention scores)
         *
         * C = Softmax(A @ B^T / sqrt(k) + mask)
         *
         * @param A Left activation matrix [m, k]
         * @param B Right activation matrix [n, k] (transposed)
         * @param C Output matrix [m, n]
         * @param m Number of rows in A and C
         * @param n Number of rows in B
         * @param k Number of columns in A and B
         * @param transpose_B Whether to transpose B
         * @param softmax_axis Axis for softmax
         * @param mpi_ctx MPI context
         * @param device_idx Device index
         *
         * @return true on success, false on error
         */
        virtual bool multiply_with_softmax(
            const float *A, const float *B, float *C,
            int m, int n, int k,
            bool transpose_B = true,
            int softmax_axis = 1,
            const float *mask = nullptr,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)A;
            (void)B;
            (void)C;
            (void)m;
            (void)n;
            (void)k;
            (void)transpose_B;
            (void)softmax_axis;
            (void)mask;
            (void)mpi_ctx;
            (void)device_idx;
            return false;
        }

        /**
         * @brief Type-erased implementation for typed fused matmul+softmax
         *
         * @param A Left activation matrix (void* to support float/uint16_t/int8_t)
         * @param B Right activation matrix (void* to support float/uint16_t/int8_t)
         * @param C Output matrix (always float*)
         * @param m Number of rows in A and C
         * @param n Number of rows in B
         * @param k Number of columns in A and B
         * @param transpose_B Whether to transpose B
         * @param softmax_axis Axis for softmax
         * @param mask Optional mask to add to logits before softmax (broadcastable to [m, n])
         * @param mpi_ctx MPI context
         * @param device_idx Device index
         * @param format_A Format of A (FP32, BF16, FP16, INT8)
         * @param format_B Format of B (FP32, BF16, FP16, INT8)
         *
         * @return true on success, false on error
         */
        virtual bool multiply_with_softmax_typed_impl(
            const void *A, const void *B, float *C,
            int m, int n, int k,
            float scale,
            bool transpose_B,
            int softmax_axis,
            const float *mask,
            bool is_causal,
            const MPIContext *mpi_ctx,
            int device_idx,
            ActivationFormat format_A,
            ActivationFormat format_B)
        {
            (void)A;
            (void)B;
            (void)C;
            (void)m;
            (void)n;
            (void)k;
            (void)scale;
            (void)transpose_B;
            (void)softmax_axis;
            (void)mask;
            (void)is_causal;
            (void)mpi_ctx;
            (void)device_idx;
            (void)format_A;
            (void)format_B;
            return false;
        }

        /**
         * @brief Template-based typed fused matmul+softmax
         *
         * C = Softmax(A @ B^T / sqrt(k) + mask)
         *
         * @tparam ActT Activation element type (float, uint16_t, int8_t)
         * @tparam WeightT Weight element type (same options as ActT)
         *
         * @param A Left activation matrix [m, k]
         * @param B Right activation matrix [n, k] (transposed)
         * @param C Output matrix [m, n]
         * @param m Number of rows in A and C
         * @param n Number of rows in B
         * @param k Number of columns in A and B
         * @param transpose_B Whether to transpose B
         * @param softmax_axis Axis for softmax
         * @param mask Optional mask to add to logits before softmax
         * @param mpi_ctx MPI context
         * @param device_idx Device index
         * @param format Activation format hint (e.g. BF16 vs FP16 for uint16_t)
         *
         * @return true on success, false on error
         */
        template <typename ActT, typename WeightT>
        bool multiply_with_softmax_typed(
            const ActT *A, const WeightT *B, float *C,
            int m, int n, int k,
            float scale = 1.0f,
            bool transpose_B = true,
            int softmax_axis = 1,
            const float *mask = nullptr,
            bool is_causal = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            ActivationFormat format = ActivationFormat::FP32)
        {
            ActivationFormat fmt_A = (std::is_same_v<ActT, float>) ? ActivationFormat::FP32 : format;
            ActivationFormat fmt_B = (std::is_same_v<WeightT, float>) ? ActivationFormat::FP32 : format;
            return multiply_with_softmax_typed_impl(
                static_cast<const void *>(A), static_cast<const void *>(B), C,
                m, n, k, scale, transpose_B, softmax_axis, mask, is_causal, mpi_ctx, device_idx, fmt_A, fmt_B);
        }

        /**
         * @brief Type-erased implementation for typed fused matmul+softmax with striding
         */
        virtual bool multiply_with_softmax_strided_typed_impl(
            const void *A, const void *B, float *C,
            int m, int n, int k,
            int lda, int ldb, int ldc,
            float scale,
            bool transpose_B,
            int softmax_axis,
            const float *mask,
            bool is_causal,
            const MPIContext *mpi_ctx,
            int device_idx,
            ActivationFormat format_A,
            ActivationFormat format_B)
        {
            (void)A;
            (void)B;
            (void)C;
            (void)m;
            (void)n;
            (void)k;
            (void)lda;
            (void)ldb;
            (void)ldc;
            (void)scale;
            (void)transpose_B;
            (void)softmax_axis;
            (void)mask;
            (void)is_causal;
            (void)mpi_ctx;
            (void)device_idx;
            (void)format_A;
            (void)format_B;
            return false;
        }

        /**
         * @brief Template-based typed fused matmul+softmax with striding
         *
         * C = Softmax(A @ B^T / sqrt(k) + mask)
         *
         * @tparam ActT Activation element type (float, uint16_t, int8_t)
         * @tparam WeightT Weight element type (same options as ActT)
         *
         * @param A Left activation matrix [m, k]
         * @param B Right activation matrix [n, k] (transposed)
         * @param C Output matrix [m, n]
         * @param m Number of rows in A and C
         * @param n Number of rows in B
         * @param k Number of columns in A and B
         * @param transpose_B Whether to transpose B
         * @param softmax_axis Axis for softmax
         * @param mask Optional mask to add to logits before softmax
         * @param mpi_ctx MPI context
         * @param device_idx Device index
         * @param format Activation format hint (e.g. BF16 vs FP16 for uint16_t)
         *
         * @return true on success, false on error
         */
        template <typename ActT, typename WeightT>
        bool multiply_with_softmax_strided_typed(
            const ActT *A, const WeightT *B, float *C,
            int m, int n, int k,
            int lda, int ldb, int ldc,
            float scale = 1.0f,
            bool transpose_B = true,
            int softmax_axis = 1,
            const float *mask = nullptr,
            bool is_causal = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            ActivationFormat format = ActivationFormat::FP32)
        {
            ActivationFormat fmt_A = (std::is_same_v<ActT, float>) ? ActivationFormat::FP32 : format;
            ActivationFormat fmt_B = (std::is_same_v<WeightT, float>) ? ActivationFormat::FP32 : format;
            return multiply_with_softmax_strided_typed_impl(
                static_cast<const void *>(A), static_cast<const void *>(B), C,
                m, n, k, lda, ldb, ldc, scale, transpose_B, softmax_axis, mask, is_causal, mpi_ctx, device_idx, fmt_A, fmt_B);
        }

        // =============================================================================
        // Fused Multi-GEMM Interface (Activation Sharing)
        // =============================================================================
        // These methods enable fusing multiple GEMMs that share the same input activations.
        // For quantized kernels, this avoids redundant FP32→Q8_1 quantization passes.
        //
        // Use cases:
        // - FFN: gate + up projections share same input (FusedDualGEMM)
        // - Attention: Q + K + V projections share same input (FusedTripleGEMM)
        //
        // The workflow:
        // 1. Call quantize_activations() once to get reusable Q8_1 buffer
        // 2. Call multiply_with_precomputed_q8_1() N times, passing the same buffer
        // 3. Buffer can be a thread-local scratch to avoid allocation
        // =============================================================================

        /**
         * @brief Quantize FP32 activations to Q8_1 format for reuse across multiple GEMMs
         *
         * This method quantizes FP32 activations to Q8_1 blocks that can be reused
         * by multiple subsequent GEMM operations. This is the first step in the
         * fused multi-GEMM workflow.
         *
         * @param A Input FP32 activations [m, k]
         * @param q8_1_buffer Output buffer for Q8_1 blocks [m * k_blocks], where k_blocks = k/32
         *                    Must be pre-allocated with at least m * ((k+31)/32) * sizeof(Q8_1Block) bytes
         * @param m Number of rows in activation matrix
         * @param k Number of columns in activation matrix
         *
         * @return true on success, false if not supported (e.g., non-quantized kernel)
         *
         * @note This is a no-op for floating-point kernels that don't need quantization.
         *       For QuantisedGemmKernel, this performs the FP32→Q8_1 conversion.
         *
         * @note Thread safety: This method is thread-safe. The output buffer can be
         *       thread-local to avoid contention.
         */
        virtual bool quantize_activations(
            const float *A,
            void *q8_1_buffer,
            int m, int k)
        {
            (void)A;
            (void)q8_1_buffer;
            (void)m;
            (void)k;
            return false; // Not implemented for non-quantized kernels
        }

        /**
         * @brief Get required buffer size for quantized activations
         *
         * @param m Number of rows in activation matrix
         * @param k Number of columns in activation matrix
         * @return Required buffer size in bytes, or 0 if quantization not supported
         */
        virtual size_t get_quantized_activation_buffer_size(int m, int k) const
        {
            (void)m;
            (void)k;
            return 0; // Not supported for non-quantized kernels
        }

        /**
         * @brief GEMM with pre-quantized activations (no redundant quantization)
         *
         * This method performs GEMM using pre-quantized Q8_1 activations from a prior
         * quantize_activations() call. This is the second step in the fused multi-GEMM
         * workflow and enables activation sharing across multiple GEMMs.
         *
         * Performance: When calling N GEMMs with the same input, this saves (N-1)
         * quantization passes compared to calling multiply() N times.
         *
         * @param q8_1_activations Pre-quantized Q8_1 blocks from quantize_activations()
         * @param C Output matrix [m, n] (FP32)
         * @param m Number of rows in activation matrix
         * @param n Number of columns in output (rows in weight matrix)
         * @param k Number of columns in activation matrix
         * @param bias Optional bias vector [n] to add after GEMM (nullptr = no bias)
         * @param accumulate If true, add to existing C; if false, overwrite
         * @param alpha Scale factor for GEMM result
         * @param beta Scale factor for existing C (only used if accumulate=true)
         * @param mpi_ctx MPI context for distributed execution
         * @param device_idx Device index (-1 = CPU)
         *
         * @return true on success, false if not supported
         *
         * @note For non-quantized kernels, this returns false. Use multiply() instead.
         * @note The q8_1_activations buffer must have been populated by quantize_activations()
         *       with the same m and k dimensions.
         */
        virtual bool multiply_with_precomputed_q8_1(
            const void *q8_1_activations,
            float *C,
            int m, int n, int k,
            const float *bias = nullptr,
            bool accumulate = false,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            // Delegate to fused version with no fusion
            return multiply_with_precomputed_q8_1(
                q8_1_activations, C, m, n, k, bias, accumulate, alpha, beta,
                mpi_ctx, device_idx, GemmFusedOps::none());
        }

        /**
         * @brief GEMM with pre-quantized activations and fused post-ops
         *
         * Extended version that supports fused operations (SwiGLU, Softmax, etc.)
         * after the GEMM computation. This enables single-pass execution of
         * common patterns like FFN (GEMM + SwiGLU) and attention (GEMM + Softmax).
         *
         * @param q8_1_activations Pre-quantized Q8_1 blocks from quantize_activations()
         * @param C Output matrix [m, n] (FP32)
         * @param m Number of rows in activation matrix
         * @param n Number of columns in output (rows in weight matrix)
         * @param k Number of columns in activation matrix
         * @param bias Optional bias vector [n] to add after GEMM (nullptr = no bias)
         * @param accumulate If true, add to existing C; if false, overwrite
         * @param alpha Scale factor for GEMM result
         * @param beta Scale factor for existing C (only used if accumulate=true)
         * @param mpi_ctx MPI context for distributed execution
         * @param device_idx Device index (-1 = CPU)
         * @param fused_ops Fused operation configuration (SwiGLU, Softmax, etc.)
         *
         * @return true on success, false if not supported
         *
         * Example - SwiGLU fusion for FFN:
         * @code
         * // Compute gate projection first
         * gate_kernel->multiply_with_precomputed_q8_1(q8_acts, gate_out, m, n, k);
         *
         * // Compute up projection with fused SwiGLU
         * up_kernel->multiply_with_precomputed_q8_1(q8_acts, up_out, m, n, k,
         *     nullptr, false, 1.0f, 0.0f, nullptr, -1,
         *     GemmFusedOps::swiglu(gate_out));
         * @endcode
         */
        virtual bool multiply_with_precomputed_q8_1(
            const void *q8_1_activations,
            float *C,
            int m, int n, int k,
            const float *bias,
            bool accumulate,
            float alpha, float beta,
            const MPIContext *mpi_ctx,
            int device_idx,
            const GemmFusedOps &fused_ops)
        {
            (void)q8_1_activations;
            (void)C;
            (void)m;
            (void)n;
            (void)k;
            (void)bias;
            (void)accumulate;
            (void)alpha;
            (void)beta;
            (void)mpi_ctx;
            (void)device_idx;
            (void)fused_ops;
            return false; // Not implemented for non-quantized kernels
        }

        /**
         * @brief Check if this kernel supports the fused multi-GEMM interface
         *
         * @return true if quantize_activations() and multiply_with_precomputed_q8_1()
         *         are implemented
         */
        virtual bool supports_activation_sharing() const
        {
            return false; // Default: not supported
        }
    };

    class TensorBase;

    /**
     * @brief Attention computation kernel
     */
    class ITensorAttention : public ITensorKernel
    {
    public:
        virtual bool compute(
            const float *Q, const float *K, const float *V, float *output,
            int batch_size, int num_heads, int seq_len, int head_dim,
            bool is_causal, int rot_offset,
            TensorBase *k_cache, TensorBase *v_cache,
            TensorBase *k_rope, TensorBase *v_rope,
            bool use_cache,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;

        virtual bool compute_batch(
            const float *Q, const float *K, const float *V, float *output,
            int batch_size, int num_heads, int seq_len, int head_dim,
            int kv_seq_len,
            bool is_causal, int rot_offset,
            TensorBase *k_cache, TensorBase *v_cache,
            TensorBase *k_rope, TensorBase *v_rope,
            bool use_cache,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;
    };

    /**
     * @brief Rotary Positional Embedding (RoPE) kernel
     */
    class ITensorRoPE : public ITensorKernel
    {
    public:
        virtual bool apply(
            float *data, float *output,
            const int *pos_ids,
            int batch_size, int seq_len, int head_dim, int num_heads,
            float theta_base, bool interleaved,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;

        virtual bool apply_bf16(
            uint16_t *data, uint16_t *output,
            const int *pos_ids,
            int batch_size, int seq_len, int head_dim, int num_heads,
            float theta_base, int device_idx) { return false; }

        virtual bool apply_fp16(
            uint16_t *data, uint16_t *output,
            const int *pos_ids,
            int batch_size, int seq_len, int head_dim, int num_heads,
            float theta_base, int device_idx) { return false; }

        virtual bool apply_q8_1(
            void *data, void *output,
            const int *pos_ids,
            int batch_size, int seq_len, int head_dim, int num_heads,
            float theta_base, int device_idx) { return false; }
    };

    /**
     * @brief SwiGLU activation kernel
     */
    class ITensorSwiGLU : public ITensorKernel
    {
    public:
        virtual bool apply(
            const float *gate, const float *up, float *output,
            int rows, int cols,
            bool add_residual,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;

        virtual bool apply_bf16(
            const uint16_t *gate, const uint16_t *up, uint16_t *output,
            int rows, int cols,
            bool add_residual,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) { return false; }

        virtual bool apply_fp16(
            const uint16_t *gate, const uint16_t *up, uint16_t *output,
            int rows, int cols,
            bool add_residual,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) { return false; }

        virtual bool apply_q8_1(
            const void *gate, const void *up, void *output,
            int rows, int cols,
            bool add_residual,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) { return false; }
    };

    /**
     * @brief Softmax kernel
     */
    class ITensorSoftmax : public ITensorKernel
    {
    public:
        virtual bool apply(
            const float *input, float *output,
            int rows, int cols,
            bool use_causal_mask,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;

        virtual bool apply_bf16(
            const uint16_t *input, uint16_t *output,
            int rows, int cols,
            bool use_causal_mask,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) { return false; }

        virtual bool apply_fp16(
            const uint16_t *input, uint16_t *output,
            int rows, int cols,
            bool use_causal_mask,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) { return false; }
    };

    /**
     * @brief RMS Normalization kernel
     */
    class ITensorRMSNorm : public ITensorKernel
    {
    public:
        virtual bool apply(
            const float *input, const float *weight, float *output,
            int rows, int cols,
            float epsilon = 1e-6f,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;

        virtual bool apply_bf16(
            const uint16_t *input, const float *weight, uint16_t *output,
            int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) { return false; }

        virtual bool apply_fp16(
            const uint16_t *input, const float *weight, uint16_t *output,
            int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) { return false; }

        virtual bool apply_int32_to_int8(
            const int32_t *input, const float *weight, int8_t *output,
            float *scales, int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) { return false; }

        /**
         * @brief Execute fused RMSNorm + INT8 quantization (operator fusion API)
         *
         * Fuses RMS normalization with per-row INT8 quantization in a single pass.
         * Eliminates intermediate FP32 buffer and redundant memory traffic.
         *
         * Algorithm:
         * 1. Compute RMS per row: rms = sqrt(mean(x²) + eps)
         * 2. Normalize: x_norm = x / rms
         * 3. Apply gamma: x_scaled = x_norm * gamma
         * 4. Quantize per-row: x_int8 = round(x_scaled / scale), scale = max(|x_scaled|) / 127
         *
         * Performance Benefits:
         * - Saves 1 FP32 intermediate buffer allocation (rows × cols × 4 bytes)
         * - Reduces memory bandwidth by ~33% (1 write instead of 2)
         * - Improves cache efficiency (single-pass algorithm)
         * - Expected speedup: 5-10% per RMSNorm operation
         *
         * @param input Input tensor [rows, cols] FP32
         * @param weight RMSNorm scale parameters [cols] FP32 (gamma)
         * @param output Output tensor [rows, cols] INT8
         * @param scales Per-row quantization scales [rows] FP32 (output parameter)
         * @param rows Number of rows (sequence length)
         * @param cols Hidden dimension (d_model)
         * @param epsilon RMSNorm epsilon for numerical stability (default: 1e-6)
         * @param mpi_ctx MPI context for distributed execution (nullptr = single node)
         * @param device_idx Device index for execution (-1 = host/CPU, ≥0 = GPU)
         *
         * @return true on success, false if not supported by this kernel
         *
         * @note Default implementation returns false (not implemented).
         *       FusedRMSNormQuantize kernel overrides this for optimized fused path.
         *       Standard RMSNorm kernels do not implement this (use apply() instead).
         */
        virtual bool execute(
            const float *input,
            const float *weight,
            int8_t *output,
            float *scales,
            int rows,
            int cols,
            float epsilon = 1e-6f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)input;
            (void)weight;
            (void)output;
            (void)scales;
            (void)rows;
            (void)cols;
            (void)epsilon;
            (void)mpi_ctx;
            (void)device_idx;
            return false; // Not implemented by default (use apply() for standard RMSNorm)
        }
    };

} // namespace llaminar2
