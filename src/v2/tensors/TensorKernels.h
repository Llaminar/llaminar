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
#include "KernelSnapshotInfo.h"
#include <memory>
#include <vector>

namespace llaminar2
{
    // Forward declarations
    class TensorBase;
    struct Q8_1Block; // For apply_q8_1() interface

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
     *
     * All kernels inherit from ITensorKernel, which in turn inherits from
     * IKernelSnapshotCapable. This enforces that all kernel implementations
     * MUST implement getKernelSnapshotInfo() for snapshot support.
     *
     * @note The pure virtual getKernelSnapshotInfo() method is inherited
     *       from IKernelSnapshotCapable. Subclasses MUST override it or
     *       the code will not compile.
     */
    class ITensorKernel : public IKernelSnapshotCapable
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
         * @brief Tensor-based matrix multiplication with explicit dimensions
         *
         * This overload accepts explicit m, n, k dimensions, essential for:
         * - **Pre-allocated buffers**: Tensor shapes may be larger than actual data
         * - **Row-parallel GEMM**: Local output size differs from tensor allocation
         * - **Batched execution**: Processing subset of pre-allocated batch buffer
         *
         * @param A Input activations tensor [>=m, >=k] (actual data in first m rows)
         * @param C Output tensor [>=m, >=n] (writes to first m rows)
         * @param m Number of rows to process (may be < A->shape()[0])
         * @param n Number of output columns (may be < C->shape()[1])
         * @param k Number of input columns (may be < A->shape()[1])
         * @param transpose_B Whether B (packed weights) is transposed (typical: true)
         * @param alpha Scale factor for A@B
         * @param beta Scale factor for existing C (for fused add)
         * @param mpi_ctx MPI context for distributed execution
         * @param device_idx Device index for execution
         *
         * @return true on success, false if format combination not supported
         */
        virtual bool multiply_tensor(
            const TensorBase *A, TensorBase *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            // Default: not supported
            (void)A;
            (void)C;
            (void)m;
            (void)n;
            (void)k;
            (void)transpose_B;
            (void)alpha;
            (void)beta;
            (void)mpi_ctx;
            (void)device_idx;
            return false;
        }

        /**
         * @brief GEMM with FP32 input, outputting to Q8_1 format
         *
         * Performs GEMM and requantizes the output directly to Q8_1 blocks.
         * This is more efficient than separate GEMM + quantization passes.
         *
         * @param A FP32 activation matrix [m, k] (will be quantized internally)
         * @param C_q8_1 Output Q8_1 blocks buffer [m, (n+31)/32 blocks]
         * @param m Number of rows (sequence length)
         * @param n Output features (columns)
         * @param k Input features
         * @param mpi_ctx MPI context for distributed execution
         * @param device_idx Device index for execution
         *
         * @return true on success, false if not supported
         *
         * @note Default implementation returns false (not supported).
         *       Kernels like QuantisedGemmKernel override with optimized implementation.
         */
        virtual bool multiply_to_q8_1(
            const float *A,
            void *C_q8_1,
            int m, int n, int k,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            // Default: not supported
            (void)A;
            (void)C_q8_1;
            (void)m;
            (void)n;
            (void)k;
            (void)mpi_ctx;
            (void)device_idx;
            return false;
        }

        /**
         * @brief Descriptor for fused multi-projection GEMM
         *
         * Used by multiply_fused() to specify multiple output projections
         * that share the same input. This is device-agnostic - the kernel
         * implementation determines how to optimize (e.g., CPU quantizes once,
         * GPU might batch into single kernel launch).
         */
        struct FusedProjectionDesc
        {
            ITensorGemm *kernel;               ///< GEMM kernel for this projection (with packed weights)
            float *output;                     ///< Output buffer [m, n]
            int n;                             ///< Output dimension (columns)
            const float *bias = nullptr;       ///< Optional bias [n]
            const float *gate_input = nullptr; ///< Optional gate for SwiGLU [m, n]
            bool do_swiglu = false;            ///< Whether to apply SwiGLU fusion
            const char *name = nullptr;        ///< Name for debug logging
        };

        /**
         * @brief Fused multi-projection GEMM: run multiple GEMMs sharing the same input
         *
         * This is optimal for patterns like Q/K/V projections or gate/up FFN projections,
         * where the same input is projected through multiple weight matrices.
         *
         * **Performance Benefits** (implementation-dependent):
         * - CPU (quantized): Input quantized once, all projections use shared Q8_1 buffer
         * - CPU (floating-point): May benefit from cache locality
         * - GPU: May batch into single kernel launch
         *
         * **Default Behavior**: Falls back to calling multiply() for each projection sequentially.
         * Subclasses can override for optimized implementations.
         *
         * @param input FP32 input activations [m, k]
         * @param projections Vector of projection descriptors
         * @param m Number of rows (batch_size * seq_len)
         * @param k Input dimension (must match all kernels' K dimension)
         * @param mpi_ctx MPI context for distributed execution
         * @param device_idx Device index for execution
         *
         * @return true on success, false on error
         */
        virtual bool multiply_fused(
            const float *input,
            const std::vector<FusedProjectionDesc> &projections,
            int m, int k,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            // Default implementation: run each projection separately
            // Subclasses (e.g., QuantisedGemmKernel) can override for optimized fusion
            for (const auto &proj : projections)
            {
                if (!proj.kernel)
                {
                    return false;
                }
                // Use the projection's kernel to run the GEMM
                bool success = proj.kernel->multiply(
                    input, proj.output,
                    m, proj.n, k,
                    true, // transpose_B (typical for weights)
                    1.0f, // alpha
                    0.0f, // beta
                    mpi_ctx,
                    device_idx);
                if (!success)
                {
                    return false;
                }
                // Note: bias and SwiGLU fusion are not handled in default implementation
                // Optimized implementations should handle these in fused manner
                if (proj.bias)
                {
                    // Simple bias add (not fused)
                    for (int i = 0; i < m; ++i)
                    {
                        for (int j = 0; j < proj.n; ++j)
                        {
                            proj.output[i * proj.n + j] += proj.bias[j];
                        }
                    }
                }
            }
            return true;
        }

        /**
         * @brief Check if this kernel supports optimized fused multi-projection
         *
         * @return true if multiply_fused() has optimized implementation beyond sequential GEMMs
         */
        virtual bool supports_fused_projection() const
        {
            return false; // Default: no optimized fusion, uses sequential fallback
        }

        /**
         * @brief GEMM with pre-quantized Q8_1 activations, outputting Q8_1
         *
         * This method takes Q8_1 blocks as input and produces Q8_1 blocks as output,
         * avoiding double quantization that would occur if using FP32 intermediate values.
         *
         * @param q8_1_activations Pre-quantized Q8_1 blocks [m, k/32]
         * @param C_q8_1 Output Q8_1 blocks buffer [m, (n+31)/32 blocks]
         * @param m Number of rows (sequence length)
         * @param n Output features (columns)
         * @param k Input features
         * @param bias Optional bias vector [n] to add before requantization (nullptr = no bias)
         * @param accumulate Unused (Q8_1 output doesn't support accumulation)
         * @param mpi_ctx MPI context for distributed execution
         * @param device_idx Device index for execution
         *
         * @return true on success, false if not supported
         */
        virtual bool multiply_with_precomputed_q8_1_to_q8_1(
            const void *q8_1_activations,
            void *C_q8_1,
            int m, int n, int k,
            const float *bias = nullptr,
            bool accumulate = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            // Default: not supported
            (void)q8_1_activations;
            (void)C_q8_1;
            (void)m;
            (void)n;
            (void)k;
            (void)bias;
            (void)accumulate;
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
         * @brief Tensor-based activation-activation matmul: C = alpha * A @ B^T + beta * C
         *
         * Type-aware dispatch based on tensor native_type():
         * - FP32 × FP32 → FP32: Standard OpenBLAS/OneDNN path
         * - BF16 × BF16 → FP32: OneDNN bf16bf16f32
         * - FP16 × FP16 → FP32: OneDNN fp16fp16f32
         * - Q8_1 × Q8_1 → FP32: Dequant both, FP32 matmul (attention scores are always FP32)
         *
         * @param A Left activation tensor [m, k]
         * @param B Right activation tensor [n, k] if transpose_B, [k, n] otherwise
         * @param C Output tensor [m, n]
         * @param transpose_B Whether to transpose B (true for Q@K^T)
         * @param alpha Scale factor (e.g., 1/sqrt(d_k) for attention)
         * @param beta Scale factor for existing C
         * @param mpi_ctx MPI context (nullptr = single node)
         * @param device_idx Device index (-1 = CPU, ≥0 = GPU)
         *
         * @return true on success, false if type combination not supported
         *
         * @note Default implementation returns false.
         *       Subclasses that include Tensors.h can override with type-aware dispatch.
         */
        virtual bool multiply_activations_tensor(
            const TensorBase *A, const TensorBase *B, TensorBase *C,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            // Default: not supported (TensorBase is only forward-declared here)
            // Subclasses that include Tensors.h can override with real implementation
            (void)A;
            (void)B;
            (void)C;
            (void)transpose_B;
            (void)alpha;
            (void)beta;
            (void)mpi_ctx;
            (void)device_idx;
            return false;
        }

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

        // =============================================================================
        // Weight Dimension Accessors (for Tensor Parallelism)
        // =============================================================================
        // These methods expose the actual dimensions of the packed weight matrix.
        // Essential for tensor parallelism where weights may be sharded:
        // - Column-parallel: N dimension is split across ranks
        // - Row-parallel: K dimension is split across ranks
        //
        // Callers can query these to determine actual local dimensions when weights
        // have been sharded, rather than assuming the full model dimensions.
        // =============================================================================

        /**
         * @brief Get the output dimension (N) of the weight matrix
         *
         * For sharded column-parallel weights (QKV, Gate/Up), this returns the
         * local N dimension (n_full / world_size), not the full model dimension.
         *
         * @return Number of output features this kernel produces, or 0 if unknown
         */
        virtual int get_n() const { return 0; }

        /**
         * @brief Get the input dimension (K) of the weight matrix
         *
         * For sharded row-parallel weights (Wo, Down), this returns the
         * local K dimension (k_full / world_size), not the full model dimension.
         *
         * @return Number of input features this kernel expects, or 0 if unknown
         */
        virtual int get_k() const { return 0; }
    };

    class TensorBase;

    // ==========================================================================
    // Attention Mode Detection
    // ==========================================================================

    /**
     * @brief Attention computation mode for kernel dispatch
     *
     * Different attention scenarios require different kernel implementations
     * for optimal performance:
     *
     * - **PREFILL**: Processing a full prompt (GEMM-based, high parallelism)
     * - **DECODE**: Single new token attending to cache (dot product + softmax)
     * - **BATCHED_DECODE**: Multiple sequences, each decoding one token
     * - **CHUNKED_PREFILL**: Large prompt split into chunks (for memory limits)
     *
     * Kernels use this enum for internal dispatch to specialized implementations.
     */
    enum class AttentionMode
    {
        PREFILL,        ///< seq_len > 1, kv_len == seq_len (batch GEMM path)
        DECODE,         ///< seq_len == 1, single sequence (dot product path)
        BATCHED_DECODE, ///< batch_size > 1, seq_len == 1 (parallel decode)
        CHUNKED_PREFILL ///< seq_len > 1, kv_len > seq_len (incremental prefill)
    };

    /**
     * @brief Detect attention mode from dimensions
     *
     * Used by kernels to dispatch to optimized implementations.
     *
     * @param batch_size Number of sequences
     * @param seq_len Query sequence length (new tokens)
     * @param kv_len Key/Value length (total cached + new tokens)
     * @return Detected AttentionMode
     *
     * Logic:
     * - batch_size > 1 && seq_len == 1 → BATCHED_DECODE
     * - seq_len == 1 → DECODE
     * - seq_len < kv_len → CHUNKED_PREFILL
     * - otherwise → PREFILL
     */
    inline AttentionMode detect_attention_mode(int batch_size, int seq_len, int kv_len)
    {
        if (batch_size > 1 && seq_len == 1)
            return AttentionMode::BATCHED_DECODE;
        if (seq_len == 1)
            return AttentionMode::DECODE;
        if (seq_len < kv_len)
            return AttentionMode::CHUNKED_PREFILL;
        return AttentionMode::PREFILL;
    }

    /**
     * @brief Get string name for AttentionMode (for logging)
     */
    inline const char *attention_mode_name(AttentionMode mode)
    {
        switch (mode)
        {
        case AttentionMode::PREFILL:
            return "PREFILL";
        case AttentionMode::DECODE:
            return "DECODE";
        case AttentionMode::BATCHED_DECODE:
            return "BATCHED_DECODE";
        case AttentionMode::CHUNKED_PREFILL:
            return "CHUNKED_PREFILL";
        default:
            return "UNKNOWN";
        }
    }

    /**
     * @brief Attention computation kernel interface
     *
     * Computes scaled dot-product attention with optional causal masking:
     *   output = softmax(Q @ K^T / sqrt(d_k) + mask) @ V
     *
     * Supports:
     * - Grouped Query Attention (GQA): n_heads != n_kv_heads
     * - Multi-Query Attention (MQA): n_kv_heads == 1
     * - Causal masking with optional sliding window
     * - Multiple activation precisions via typed implementations
     *
     * Implementations:
     * - CPUAttentionKernelT<Precision>: Precision-enum based kernel
     */
    class ITensorAttention : public ITensorKernel
    {
    public:
        /**
         * @brief Compute single-sequence attention
         *
         * @param Q Query tensor [seq_len, n_heads, head_dim] (float* for interface, cast internally)
         * @param K Key tensor [seq_len, n_kv_heads, head_dim]
         * @param V Value tensor [seq_len, n_kv_heads, head_dim]
         * @param output Output tensor [seq_len, n_heads, head_dim]
         * @param seq_len Sequence length (number of tokens)
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads (GQA: n_heads % n_kv_heads == 0)
         * @param head_dim Dimension per head
         * @param causal Apply causal (lower-triangular) masking
         * @param window_size Sliding window size (-1 = disabled)
         * @param workspace_scores Workspace for attention scores [n_heads, seq_len, seq_len]
         * @param workspace_buffer Workspace for intermediate buffers (unused with Virtual GQA)
         * @param workspace_context Workspace for context output (optional)
         * @param workspace_mask Pre-built attention mask [seq_len, seq_len] or nullptr
         * @param use_bf16 Hint for BF16 precision path (deprecated, use typed kernel)
         * @param mpi_ctx MPI context for distributed execution
         * @param device_idx Device index (-1 = CPU)
         * @return true on success
         */
        virtual bool compute(
            const float *Q, const float *K, const float *V, float *output,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal = false,
            int window_size = -1,
            TensorBase *workspace_scores = nullptr,
            TensorBase *workspace_buffer = nullptr,
            TensorBase *workspace_context = nullptr,
            TensorBase *workspace_mask = nullptr,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;

        /**
         * @brief Compute batched attention
         *
         * @param Q Query tensor [batch_size * seq_len, n_heads, head_dim]
         * @param K Key tensor [batch_size * seq_len, n_kv_heads, head_dim]
         * @param V Value tensor [batch_size * seq_len, n_kv_heads, head_dim]
         * @param output Output tensor [batch_size * seq_len, n_heads, head_dim]
         * @param batch_size Number of sequences in batch
         * @param seq_len Sequence length per batch item
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads
         * @param head_dim Dimension per head
         * @param causal Apply causal masking
         * @param window_size Sliding window size (-1 = disabled)
         * @param workspace_scores Workspace for attention scores
         * @param workspace_buffer Workspace for intermediate buffers
         * @param workspace_context Workspace for context output
         * @param workspace_mask Pre-built attention mask or nullptr
         * @param use_bf16 Hint for BF16 precision path
         * @param mpi_ctx MPI context
         * @param device_idx Device index
         * @return true on success
         */
        virtual bool compute_batch(
            const float *Q, const float *K, const float *V, float *output,
            int batch_size, int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal = false,
            int window_size = -1,
            TensorBase *workspace_scores = nullptr,
            TensorBase *workspace_buffer = nullptr,
            TensorBase *workspace_context = nullptr,
            TensorBase *workspace_mask = nullptr,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;

        /**
         * @brief Compute single-sequence Q8_1 native attention
         *
         * Native Q8_1 path that avoids float* casts. Implementations that don't
         * support Q8_1 should return false (default behavior).
         *
         * @param Q Q8_1Block* cast to void* [seq_len, n_heads, head_dim_blocks]
         * @param K Q8_1Block* cast to void* [seq_len, n_kv_heads, head_dim_blocks]
         * @param V Q8_1Block* cast to void* [seq_len, n_kv_heads, head_dim_blocks]
         * @param output Q8_1Block* cast to void* [seq_len, n_heads, head_dim_blocks]
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads
         * @param head_dim Head dimension (must be multiple of 32 for Q8_1)
         * @param causal Apply causal masking
         * @param window_size Sliding window size (-1 = disabled)
         * @param workspace_scores Workspace for attention scores
         * @param workspace_mask Pre-built attention mask or nullptr
         * @param mpi_ctx MPI context
         * @param device_idx Device index
         * @return true on success, false if Q8_1 not supported
         */
        virtual bool compute_q8_1(
            const void *Q, const void *K, const void *V, void *output,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal = false,
            int window_size = -1,
            TensorBase *workspace_scores = nullptr,
            TensorBase *workspace_mask = nullptr,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)Q;
            (void)K;
            (void)V;
            (void)output;
            (void)seq_len;
            (void)n_heads;
            (void)n_kv_heads;
            (void)head_dim;
            (void)causal;
            (void)window_size;
            (void)workspace_scores;
            (void)workspace_mask;
            (void)mpi_ctx;
            (void)device_idx;
            return false; // Default: Q8_1 not supported
        }

        /**
         * @brief Compute batched Q8_1 native attention
         *
         * Native Q8_1 batch path. See compute_q8_1() for parameter details.
         *
         * @return true on success, false if Q8_1 not supported
         */
        virtual bool compute_batch_q8_1(
            const void *Q, const void *K, const void *V, void *output,
            int batch_size, int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal = false,
            int window_size = -1,
            TensorBase *workspace_scores = nullptr,
            TensorBase *workspace_mask = nullptr,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)Q;
            (void)K;
            (void)V;
            (void)output;
            (void)batch_size;
            (void)seq_len;
            (void)n_heads;
            (void)n_kv_heads;
            (void)head_dim;
            (void)causal;
            (void)window_size;
            (void)workspace_scores;
            (void)workspace_mask;
            (void)mpi_ctx;
            (void)device_idx;
            return false; // Default: Q8_1 not supported
        }

        /**
         * @brief Compute attention using tensor objects with automatic type dispatch
         *
         * Inspects Q/K/V/output tensor native_type() and dispatches to the appropriate
         * typed method (compute, compute_batch, compute_q8_1, compute_batch_q8_1).
         *
         * @param Q Query tensor [batch_size * seq_len, n_heads * head_dim]
         * @param K Key tensor [batch_size * kv_len, n_kv_heads * head_dim]
         * @param V Value tensor [batch_size * kv_len, n_kv_heads * head_dim]
         * @param output Output tensor [batch_size * seq_len, n_heads * head_dim]
         * @param batch_size Batch size (1 for single sequence)
         * @param seq_len Query sequence length
         * @param kv_len Key/value sequence length (may differ from seq_len in decode)
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads (GQA: n_heads % n_kv_heads == 0)
         * @param head_dim Dimension per head
         * @param causal Apply causal (lower-triangular) masking
         * @param window_size Sliding window size (-1 = disabled)
         * @param workspace_scores Workspace for attention scores
         * @param workspace_mask Pre-built attention mask or nullptr
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index
         * @param head_start First query head to compute (0-indexed, default 0)
         * @param local_n_heads Number of query heads to compute (-1 = all)
         * @param local_n_kv_heads Number of KV heads for this slice (-1 = all)
         * @return true on success, false on failure or unsupported type combination
         *
         * @note Default returns false. Subclasses should override with type-aware dispatch.
         *
         * Tensor Parallelism Usage:
         * When distributing attention across ranks, each rank computes a subset of heads:
         *   Rank 0: head_start=0, local_n_heads=n_heads/2
         *   Rank 1: head_start=n_heads/2, local_n_heads=n_heads/2
         *
         * The output tensor is indexed as output[seq, head, dim], so each rank writes
         * to output[:, head_start:head_start+local_n_heads, :]. After computation,
         * an AllGather is needed to combine results (handled by caller).
         */
        virtual bool compute_tensor(
            const TensorBase *Q,
            const TensorBase *K,
            const TensorBase *V,
            TensorBase *output,
            int batch_size,
            int seq_len,
            int kv_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            bool causal = false,
            int window_size = -1,
            TensorBase *workspace_scores = nullptr,
            TensorBase *workspace_mask = nullptr,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            int head_start = 0,        ///< First query head (TP slice start)
            int local_n_heads = -1,    ///< Number of query heads (-1 = all)
            int local_n_kv_heads = -1) ///< Number of KV heads (-1 = all)
        {
            (void)Q;
            (void)K;
            (void)V;
            (void)output;
            (void)batch_size;
            (void)seq_len;
            (void)kv_len;
            (void)n_heads;
            (void)n_kv_heads;
            (void)head_dim;
            (void)causal;
            (void)window_size;
            (void)workspace_scores;
            (void)workspace_mask;
            (void)mpi_ctx;
            (void)device_idx;
            (void)head_start;
            (void)local_n_heads;
            (void)local_n_kv_heads;
            return false; // Subclasses override with type-aware dispatch
        }
    };

    // =========================================================================
    // Forward Declarations for Fused Attention
    // =========================================================================

    namespace gemm_v4
    {
        struct QuantisedPackedWeights;
    }

    // =========================================================================
    // ITensorFusedAttentionWo - Fused Attention + Wo Projection Interface
    // =========================================================================

    /**
     * @brief Parameters for fused attention + Wo projection + residual add
     *
     * This struct bundles all parameters for a fully-fused attention block that
     * combines:
     * 1. Q×K^T → attention scores (INT32 or FP32 domain)
     * 2. Softmax → attention weights
     * 3. Weights×V → attention context
     * 4. Context×Wo → output projection
     * 5. Output + residual → final output
     *
     * Design Goals:
     * - **Register-resident intermediate values**: No round-trip to memory between steps
     * - **Integer-domain pipeline**: Q16_1 implementation keeps everything in INT32
     * - **Flexible I/O**: Supports various quantized formats (Q16_1, Q8_1, FP32)
     *
     * Execution paths (determined by implementation):
     * - **Flash Decode** (seq_len_q=1): Single query, full KV, no tiling
     * - **FA2 Prefill** (seq_len_q>1): Batched queries, tiled KV processing
     *
     * @note Q, K, V, residual types are void* to support both quantized blocks
     *       and FP32 pointers. The implementing kernel knows the actual types.
     */
    struct FusedAttentionWoParams
    {
        // ============== Attention tensors (Q16_1, Q8_1, or FP32) ==============

        const void *Q = nullptr; ///< Query tensor [seq_len_q × n_heads × head_dim]
        const void *K = nullptr; ///< Key tensor [kv_len × n_kv_heads × head_dim]
        const void *V = nullptr; ///< Value tensor [kv_len × n_kv_heads × head_dim]

        // ============== Wo projection weights ==============

        /**
         * @brief VNNI-packed Wo weights for VPDPWSSD (INT16×INT16→INT32)
         *
         * Created from Q8_1 via KernelFactory::ensurePackedWeightsInTensorCache()
         * For FP32 implementations, this may be nullptr (use FP32 Wo instead)
         */
        const gemm_v4::QuantisedPackedWeights *Wo_packed = nullptr;

        /**
         * @brief FP32 Wo weights [d_model × d_model] for FP32 implementations
         *
         * Used when Wo_packed is nullptr (pure FP32 path)
         */
        const float *Wo_fp32 = nullptr;

        // ============== Residual tensors ==============

        const void *residual_in = nullptr; ///< Input residual [seq_len_q × d_model]
        void *residual_out = nullptr;      ///< Output residual [seq_len_q × d_model]
                                           ///< Can be same as residual_in for in-place

        // ============== Dimensions ==============

        int seq_len_q = 0;  ///< Number of query positions
        int kv_len = 0;     ///< Number of KV positions (cache length)
        int n_heads = 0;    ///< Number of query heads
        int n_kv_heads = 0; ///< Number of KV heads (for GQA/MQA)
        int head_dim = 0;   ///< Dimension per head (typically 64 or 128)
        int d_model = 0;    ///< Model dimension (n_heads * head_dim)

        // ============== Attention configuration ==============

        float scale = 0.0f;      ///< 1/sqrt(head_dim), auto-computed if 0
        bool causal = true;      ///< Apply causal masking
        int position_offset = 0; ///< Position offset for causal mask (decode)
        int window_size = -1;    ///< Sliding window size (-1 = disabled)

        // ============== Tiling parameters (for FA2 prefill) ==============

        int Bc = 256; ///< KV tile size for FA2 prefill (ignored for decode)
        int Br = 16;  ///< Query tile size for FA2 prefill

        // ============== Snapshot/Debug buffers ==============

        /**
         * @brief Optional FP32 buffer for attention scores snapshot
         *
         * If non-null, attention scores (Q×K^T × scale) will be dequantized
         * to FP32 and stored here [seq_len_q × n_heads × kv_len]
         * for debugging/parity testing.
         */
        float *scores_snapshot = nullptr;

        /**
         * @brief Optional FP32 buffer for attention context snapshot
         *
         * If non-null, context (softmax × V) will be dequantized to FP32
         * and stored here [seq_len_q × n_heads × head_dim]
         * before Wo projection.
         */
        float *context_snapshot = nullptr;

        /**
         * @brief Optional FP32 buffer for Wo projection output snapshot
         *
         * If non-null, Wo output will be dequantized to FP32
         * and stored here [seq_len_q × d_model]
         * before residual add.
         */
        float *wo_output_snapshot = nullptr;

        /**
         * @brief Optional FP32 buffer for attention residual output snapshot
         *
         * If non-null, the final output (after residual add) will be
         * dequantized to FP32 and stored here [seq_len_q × d_model].
         * Corresponds to ATTENTION_RESIDUAL in the FP32 pipeline.
         */
        float *attention_residual_snapshot = nullptr;

        // ============== Optional debug metadata ==============

        /**
         * @brief Layer index for debug/snapshot correlation
         *
         * Not required for kernel correctness. When set, debug instrumentation can
         * gate dumps to a specific layer to keep logs manageable.
         */
        int layer_idx = -1;

        // ============== Helper methods ==============

        /**
         * @brief Get the attention scale (auto-compute if not set)
         */
        float get_scale() const
        {
            if (scale > 0.0f)
                return scale;
            return 1.0f / std::sqrt(static_cast<float>(head_dim));
        }

        /**
         * @brief Check if this is a decode pass (single query token)
         */
        bool is_decode() const { return seq_len_q == 1; }

        /**
         * @brief Get the KV head index for a given query head (GQA mapping)
         */
        int get_kv_head(int query_head) const
        {
            if (n_kv_heads == n_heads)
                return query_head;
            return query_head / (n_heads / n_kv_heads);
        }
    };

    /**
     * @brief Interface for fused attention + Wo projection + residual kernels
     *
     * This interface is designed for high-performance fused attention kernels
     * that combine the entire attention block (QKV → softmax → context → Wo → residual)
     * into a single kernel call, maximizing register utilization.
     *
     * Primary use case: **Integer-domain Q16_1 attention**
     * - Q×K^T → INT32 scores
     * - Exp2FixedSoftmax → INT16 weights
     * - Weights×V → INT32 context accumulators
     * - Context×Wo (VPDPWSSD) → INT32 → Q16_1
     * - Q16_1 + Q16_1 residual → Q16_1 output
     *
     * The interface also supports FP32 implementations for reference/fallback.
     *
     * **Why separate from ITensorAttention?**
     * ITensorAttention outputs attention context only. This interface outputs
     * the full attention block result (after Wo and residual), enabling:
     * - Register-resident Wo projection (no context materialize to memory)
     * - Native quantized residual add (no FP32 conversion)
     * - Single kernel launch for entire attention block
     *
     * Implementations:
     * - Q16FusedAttentionRef: Scalar C++ reference (for testing/debugging)
     * - Q16FusedAttentionJit: AVX512 JIT (production)
     * - FP32FusedAttentionRef: FP32 fallback (parity testing)
     */
    class ITensorFusedAttentionWo : public ITensorKernel
    {
    public:
        /**
         * @brief Execute fused attention + Wo + residual
         *
         * Dispatches to decode or prefill path based on params.seq_len_q.
         *
         * @param params Complete parameter struct (see FusedAttentionWoParams)
         * @param mpi_ctx MPI context for distributed execution (optional)
         * @param device_idx Device index (-1 = CPU)
         * @return true on success, false on validation/execution error
         *
         * @note This is the primary entry point. Implementations may override
         *       compute_decode() and compute_prefill() for path-specific optimizations.
         */
        virtual bool compute(
            const FusedAttentionWoParams &params,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;

        /**
         * @brief Execute fused attention for decode (single query token)
         *
         * Optimized for autoregressive token generation:
         * - Single query against full KV cache
         * - Parallel over KV positions, no tiling
         * - Latency-optimized
         *
         * @param params Parameters (must have seq_len_q == 1)
         * @param mpi_ctx MPI context
         * @param device_idx Device index
         * @return true on success
         *
         * Default: calls compute(params, mpi_ctx, device_idx) with validation
         */
        virtual bool compute_decode(
            const FusedAttentionWoParams &params,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            if (params.seq_len_q != 1)
                return false;
            return compute(params, mpi_ctx, device_idx);
        }

        /**
         * @brief Execute fused attention for prefill (multiple query tokens)
         *
         * Optimized for prompt processing:
         * - Batched queries with tiled KV processing
         * - Throughput-optimized (FA2 algorithm)
         *
         * @param params Parameters (typically seq_len_q > 1)
         * @param mpi_ctx MPI context
         * @param device_idx Device index
         * @return true on success
         *
         * Default: calls compute(params, mpi_ctx, device_idx) with validation
         */
        virtual bool compute_prefill(
            const FusedAttentionWoParams &params,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            if (params.seq_len_q < 1)
                return false;
            return compute(params, mpi_ctx, device_idx);
        }

        /**
         * @brief Execute using TensorBase objects with automatic type dispatch
         *
         * Inspects tensor native_type() and constructs appropriate params.
         * This is the type-safe interface for stages that don't know the
         * concrete tensor format.
         *
         * @param Q Query tensor [seq_len_q × n_heads × head_dim]
         * @param K Key tensor [kv_len × n_kv_heads × head_dim]
         * @param V Value tensor [kv_len × n_kv_heads × head_dim]
         * @param Wo_tensor Wo weight tensor (for extracting packed weights or FP32 data)
         * @param residual_in Input residual tensor [seq_len_q × d_model]
         * @param residual_out Output residual tensor [seq_len_q × d_model]
         * @param seq_len_q Query sequence length
         * @param kv_len KV sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of KV heads
         * @param head_dim Head dimension
         * @param causal Apply causal masking
         * @param position_offset Position offset for decode
         * @param scores_snapshot Optional FP32 buffer for scores
         * @param context_snapshot Optional FP32 buffer for context
         * @param wo_output_snapshot Optional FP32 buffer for Wo output
         * @param mpi_ctx MPI context
         * @param device_idx Device index
         * @return true on success, false if tensor types not supported
         */
        virtual bool compute_tensor(
            const TensorBase *Q,
            const TensorBase *K,
            const TensorBase *V,
            const TensorBase *Wo_tensor,
            const TensorBase *residual_in,
            TensorBase *residual_out,
            int seq_len_q,
            int kv_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            bool causal = true,
            int position_offset = 0,
            float *scores_snapshot = nullptr,
            float *context_snapshot = nullptr,
            float *wo_output_snapshot = nullptr,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            // Default: not supported (subclasses override with type-aware dispatch)
            (void)Q;
            (void)K;
            (void)V;
            (void)Wo_tensor;
            (void)residual_in;
            (void)residual_out;
            (void)seq_len_q;
            (void)kv_len;
            (void)n_heads;
            (void)n_kv_heads;
            (void)head_dim;
            (void)causal;
            (void)position_offset;
            (void)scores_snapshot;
            (void)context_snapshot;
            (void)wo_output_snapshot;
            (void)mpi_ctx;
            (void)device_idx;
            return false;
        }

        /**
         * @brief Get input precision format supported by this kernel
         *
         * Returns the activation precision this kernel expects for Q/K/V inputs.
         * Used by stages to verify tensor format compatibility.
         */
        virtual ActivationFormat input_format() const = 0;

        /**
         * @brief Get output precision format produced by this kernel
         *
         * Returns the activation precision of the residual output.
         */
        virtual ActivationFormat output_format() const = 0;

        /**
         * @brief Check if this kernel requires VNNI-packed Wo weights
         *
         * @return true if Wo_packed must be provided, false if Wo_fp32 is used
         */
        virtual bool requires_packed_wo() const = 0;
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

        /**
         * @brief Apply RoPE in-place to Q16_1 tensors
         *
         * Q16_1 has higher precision (256× finer than Q8_1) with FP32 scale,
         * making in-place rotation with intermediate requantization acceptable.
         *
         * @param Q_data Q16_1 Q tensor blocks - modified in-place
         * @param K_data Q16_1 K tensor blocks - modified in-place (can be nullptr)
         * @param pos_ids Position indices [seq_len]
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of KV heads
         * @param head_dim Head dimension
         * @param theta_base RoPE frequency base
         * @param device_idx Device index
         * @return true on success
         */
        virtual bool apply_q16_1(
            void *Q_data, void *K_data,
            const int *pos_ids,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            float theta_base, int device_idx) { return false; }

        /**
         * @brief Apply RoPE to Q8_1 input, output to FP32 (Hybrid mode)
         *
         * This method dequantizes Q8_1 input, applies rotation, and outputs
         * to FP32 WITHOUT requantization. Used in Hybrid activation precision
         * mode to eliminate the dequant→rotate→requant cycle.
         *
         * @param Q_in Q8_1 Q input tensor [seq_len, n_heads * head_dim]
         * @param K_in Q8_1 K input tensor [seq_len, n_kv_heads * head_dim] or nullptr
         * @param Q_out FP32 Q output tensor [seq_len, n_heads * head_dim]
         * @param K_out FP32 K output tensor [seq_len, n_kv_heads * head_dim] or nullptr
         * @param position_ids Position indices [seq_len]
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of KV heads
         * @param head_dim Head dimension
         * @param rope_theta RoPE frequency base
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index
         * @return true on success
         */
        virtual bool apply_q8_1_to_fp32(
            TensorBase *Q_in,
            TensorBase *K_in,
            TensorBase *Q_out,
            TensorBase *K_out,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)Q_in;
            (void)K_in;
            (void)Q_out;
            (void)K_out;
            (void)position_ids;
            (void)seq_len;
            (void)n_heads;
            (void)n_kv_heads;
            (void)head_dim;
            (void)rope_theta;
            (void)mpi_ctx;
            (void)device_idx;
            return false; // Default: not supported
        }

        /**
         * @brief Apply RoPE to Q8_1 input, output to Q16_1 (HybridQ16 mode)
         *
         * This method dequantizes Q8_1 input, applies rotation, and outputs
         * to Q16_1 (higher precision). Used in HybridQ16 activation precision
         * mode for the Q16 fused attention kernel.
         *
         * Q16_1 output has 256× finer precision than Q8_1, enabling high-accuracy
         * integer attention without FP32 intermediate storage.
         *
         * @param Q_in Q8_1 Q input tensor [seq_len, n_heads * head_dim]
         * @param K_in Q8_1 K input tensor [seq_len, n_kv_heads * head_dim] or nullptr
         * @param Q_out Q16_1 Q output tensor [seq_len, n_heads * head_dim]
         * @param K_out Q16_1 K output tensor [seq_len, n_kv_heads * head_dim] or nullptr
         * @param position_ids Position indices [seq_len]
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of KV heads
         * @param head_dim Head dimension
         * @param rope_theta RoPE frequency base
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index
         * @return true on success
         */
        virtual bool apply_q8_1_to_q16_1(
            TensorBase *Q_in,
            TensorBase *K_in,
            TensorBase *Q_out,
            TensorBase *K_out,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)Q_in;
            (void)K_in;
            (void)Q_out;
            (void)K_out;
            (void)position_ids;
            (void)seq_len;
            (void)n_heads;
            (void)n_kv_heads;
            (void)head_dim;
            (void)rope_theta;
            (void)mpi_ctx;
            (void)device_idx;
            return false; // Default: not supported
        }

        /**
         * @brief Apply RoPE to Q8_1 input, output to Q16 with per-head scale output
         *
         * Pure integer arithmetic implementation: Q8 fixed-point ratios, Q15 sin/cos,
         * integer rotation without FP32 intermediate storage. Only FP32 used is for
         * computing the output scale.
         *
         * @param Q_in Q8_1 Q input tensor [seq_len, n_heads * head_dim]
         * @param K_in Q8_1 K input tensor [seq_len, n_kv_heads * head_dim] or nullptr
         * @param Q_out Q16 Q output tensor [seq_len, n_heads * head_dim]
         * @param K_out Q16 K output tensor [seq_len, n_kv_heads * head_dim] or nullptr
         * @param Q_head_scales Output: per-head Q scales [seq_len * n_heads] or nullptr
         * @param K_head_scales Output: per-head K scales [seq_len * n_kv_heads] or nullptr
         * @param block_size Q16 output block size (32, 64, 128, or 192)
         * @param position_ids Position indices [seq_len]
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of KV heads
         * @param head_dim Head dimension
         * @param rope_theta RoPE frequency base
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index
         * @return true on success
         */
        virtual bool apply_q8_1_to_q16(
            TensorBase *Q_in,
            TensorBase *K_in,
            TensorBase *Q_out,
            TensorBase *K_out,
            float *Q_head_scales,
            float *K_head_scales,
            Q16BlockSize block_size,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)Q_in;
            (void)K_in;
            (void)Q_out;
            (void)K_out;
            (void)Q_head_scales;
            (void)K_head_scales;
            (void)block_size;
            (void)position_ids;
            (void)seq_len;
            (void)n_heads;
            (void)n_kv_heads;
            (void)head_dim;
            (void)rope_theta;
            (void)mpi_ctx;
            (void)device_idx;
            return false; // Default: not supported
        }

        /**
         * @brief Apply RoPE to Q8_1 input, output to Q16 with FIXED scale
         *
         * Pure integer arithmetic implementation with fixed output scale.
         * All output Q16 blocks have d = kv_cache_scale / 32767, matching
         * the KV cache quantization scale. This enables true integer attention
         * where Q, K, and V all use identical scales.
         *
         * Algorithm:
         * 1. Compute per-block scale ratios (O(head_dim/32) FP32 ops)
         * 2. Rescale Q8 to fixed scale (pure integer per-element)
         * 3. Apply RoPE rotation (pure integer)
         * 4. Pack to Q16 with fixed d = kv_cache_scale / 32767
         *
         * @param Q_in Q8_1 Q input tensor [seq_len, n_heads * head_dim]
         * @param K_in Q8_1 K input tensor [seq_len, n_kv_heads * head_dim] or nullptr
         * @param Q_out Q16 Q output tensor [seq_len, n_heads * head_dim]
         * @param K_out Q16 K output tensor [seq_len, n_kv_heads * head_dim] or nullptr
         * @param block_size Q16 output block size (32, 64, 128, or 192)
         * @param position_ids Position indices [seq_len]
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of KV heads
         * @param head_dim Head dimension
         * @param rope_theta RoPE frequency base
         * @param kv_cache_scale Fixed scale for all output blocks (e.g., 8.0f)
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index
         * @return true on success
         */
        virtual bool apply_q8_1_to_q16_fixed_scale(
            TensorBase *Q_in,
            TensorBase *K_in,
            TensorBase *Q_out,
            TensorBase *K_out,
            Q16BlockSize block_size,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            float kv_cache_scale,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)Q_in;
            (void)K_in;
            (void)Q_out;
            (void)K_out;
            (void)block_size;
            (void)position_ids;
            (void)seq_len;
            (void)n_heads;
            (void)n_kv_heads;
            (void)head_dim;
            (void)rope_theta;
            (void)kv_cache_scale;
            (void)mpi_ctx;
            (void)device_idx;
            return false; // Default: not supported
        }

        /**
         * @brief Apply RoPE using tensor objects with automatic type dispatch
         *
         * Inspects Q/K tensor native_type() and dispatches to the appropriate
         * typed method (apply, apply_bf16, apply_fp16, apply_q8_1).
         *
         * @param Q Query tensor [seq_len, n_heads * head_dim] - modified in-place
         * @param K Key tensor [seq_len, n_kv_heads * head_dim] - modified in-place
         * @param position_ids Position indices [seq_len]
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of KV heads
         * @param head_dim Head dimension
         * @param rope_theta RoPE frequency base
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index
         * @return true on success, false on failure or unsupported type
         *
         * @note Default returns false. Subclasses should override with type-aware dispatch.
         */
        virtual bool apply_tensor(
            TensorBase *Q,
            TensorBase *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)Q;
            (void)K;
            (void)position_ids;
            (void)seq_len;
            (void)n_heads;
            (void)n_kv_heads;
            (void)head_dim;
            (void)rope_theta;
            (void)mpi_ctx;
            (void)device_idx;
            return false; // Subclasses override with type-aware dispatch
        }
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

        /**
         * @brief Apply SwiGLU using tensor objects with automatic type dispatch
         *
         * Inspects gate/up/output tensor native_type() and dispatches to the appropriate
         * typed method (apply, apply_bf16, apply_fp16, apply_q8_1).
         *
         * @param gate Gate tensor [rows, cols] (read-only)
         * @param up Up tensor [rows, cols] (read-only)
         * @param output Output tensor [rows, cols]
         * @param rows Number of rows
         * @param cols Number of columns
         * @param add_residual Whether to add residual
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index
         * @return true on success, false on failure or unsupported type combination
         *
         * @note Default returns false. Subclasses should override with type-aware dispatch.
         */
        virtual bool apply_tensor(
            const TensorBase *gate,
            const TensorBase *up,
            TensorBase *output,
            int rows, int cols,
            bool add_residual,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) { return false; }
    };

    /**
     * @brief Embedding lookup kernel
     *
     * Handles embedding table lookup with typed output:
     * - FP32 output: Direct memcpy from embedding table
     * - BF16/FP16 output: Lookup FP32 embeddings, then convert
     * - Q8_1 output: Lookup FP32 embeddings, then quantize to Q8_1 format
     *
     * The embedding table is always FP32 (stored in model weights).
     * Output format is determined by activation precision configuration.
     */
    class ITensorEmbedding : public ITensorKernel
    {
    public:
        /**
         * @brief Execute embedding lookup with FP32 output
         *
         * @param embed_data Embedding table data [vocab_size, d_model] (FP32)
         * @param token_ids Token IDs to look up
         * @param num_tokens Number of tokens
         * @param d_model Embedding dimension
         * @param output Output buffer [num_tokens, d_model] (FP32)
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index
         * @return true on success
         */
        virtual bool apply(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            float *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;

        /**
         * @brief Execute embedding lookup with BF16 output
         */
        virtual bool apply_bf16(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            uint16_t *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) { return false; }

        /**
         * @brief Execute embedding lookup with FP16 output
         */
        virtual bool apply_fp16(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            uint16_t *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) { return false; }

        /**
         * @brief Execute embedding lookup with Q8_1 output
         *
         * @param embed_data Embedding table data [vocab_size, d_model] (FP32)
         * @param token_ids Token IDs to look up
         * @param num_tokens Number of tokens
         * @param d_model Embedding dimension (must be multiple of 32)
         * @param output Output Q8_1 blocks [num_tokens * blocks_per_row]
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index
         * @return true on success
         */
        virtual bool apply_q8_1(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            void *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) { return false; }

        /**
         * @brief Apply embedding lookup using tensor objects with automatic type dispatch
         *
         * @param embed_table Embedding table tensor [vocab_size, d_model] (FP32)
         * @param token_ids Token IDs to look up
         * @param num_tokens Number of tokens
         * @param d_model Embedding dimension
         * @param output Output tensor [num_tokens, d_model]
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index
         * @return true on success, false on failure or unsupported type
         */
        virtual bool apply_tensor(
            const TensorBase *embed_table,
            const int *token_ids,
            int num_tokens,
            int d_model,
            TensorBase *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) { return false; }
    };

    /**
     * @brief Softmax kernel
     *
     * Computes row-wise softmax normalization with optional causal masking.
     * Typically used for attention score normalization.
     *
     * The modern interface uses apply_tensor() for polymorphic dispatch.
     * Legacy typed methods (apply_bf16, apply_fp16) are retained for
     * backward compatibility but implementations should prefer apply_tensor().
     */
    class ITensorSoftmax : public ITensorKernel
    {
    public:
        /**
         * @brief Apply softmax to FP32 data (legacy interface)
         *
         * @param input Input tensor [rows, cols]
         * @param output Output tensor [rows, cols]
         * @param rows Number of rows
         * @param cols Number of columns per row
         * @param use_causal_mask Apply causal (lower-triangular) masking
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index (-1 = CPU)
         * @return true on success
         */
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

        /**
         * @brief Apply softmax using tensor objects with automatic type dispatch
         *
         * Modern polymorphic interface. Inspects input/output tensor native_type()
         * and dispatches to the appropriate precision-specific implementation.
         *
         * @param input Input tensor [rows, cols] (any supported precision)
         * @param output Output tensor [rows, cols] (same type as input, may be same buffer for in-place)
         * @param rows Number of rows
         * @param cols Number of columns per row
         * @param use_causal_mask Apply causal (lower-triangular) masking
         * @param scale Pre-softmax scale factor (typically 1/sqrt(d_k) for attention)
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index (-1 = CPU)
         * @return true on success, false on failure or unsupported type
         *
         * @note Default returns false. Subclasses should override with type-aware dispatch.
         */
        virtual bool apply_tensor(
            const TensorBase *input,
            TensorBase *output,
            int rows, int cols,
            bool use_causal_mask,
            float scale = 1.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)input;
            (void)output;
            (void)rows;
            (void)cols;
            (void)use_causal_mask;
            (void)scale;
            (void)mpi_ctx;
            (void)device_idx;
            return false; // Subclasses override with type-aware dispatch
        }
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

        virtual bool apply_q8_1(
            const Q8_1Block *input, const float *weight, Q8_1Block *output,
            int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) { return false; }

        /**
         * @brief Apply RMSNorm with Q16_1 input, FP32 output
         *
         * Special case for typed residual stream: reads Q16_1, outputs FP32.
         * Used when the residual is stored in high-precision Q16_1 format but
         * downstream operations need FP32 (e.g., before attention projections).
         *
         * @param input Input Q16_1 blocks [rows * blocks_per_row]
         * @param weight Weight tensor (gamma) [cols] - always FP32
         * @param output Output FP32 tensor [rows * cols]
         * @param rows Number of rows
         * @param cols Number of columns (hidden_dim, must be multiple of 32)
         * @param epsilon RMSNorm epsilon
         * @param device_idx Device index
         * @return true on success, false if not supported
         */
        virtual bool apply_q16_1_to_fp32(
            const Q16_1Block *input, const float *weight, float *output,
            int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) { return false; }

        virtual bool apply_int32_to_int8(
            const int32_t *input, const float *weight, int8_t *output,
            float *scales, int rows, int cols, float epsilon = 1e-6f, int device_idx = -1) { return false; }

        /**
         * @brief Apply RMSNorm using tensor objects with automatic type dispatch
         *
         * Inspects input/output tensor native_type() and dispatches to the appropriate
         * typed method (apply, apply_bf16, apply_fp16, apply_q8_1).
         *
         * @param input Input tensor [rows, cols]
         * @param weight Weight tensor (gamma) [cols] - always FP32
         * @param output Output tensor [rows, cols] - must match input type
         * @param rows Number of rows
         * @param cols Number of columns
         * @param epsilon RMSNorm epsilon
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index
         * @return true on success, false on failure or unsupported type combination
         *
         * @note Default returns false. Subclasses should override with type-aware dispatch.
         */
        virtual bool apply_tensor(
            const TensorBase *input,
            const TensorBase *weight,
            TensorBase *output,
            int rows, int cols,
            float epsilon = 1e-6f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)input;
            (void)weight;
            (void)output;
            (void)rows;
            (void)cols;
            (void)epsilon;
            (void)mpi_ctx;
            (void)device_idx;
            return false; // Subclasses override with type-aware dispatch
        }

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
