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

    /**
     * @brief Block decoder strategy interface for quantized tensors (FP32 decode path)
     *
     * Provides format-specific dequantization for block-quantized formats.
     * Implementations MUST be header-only with always_inline to ensure zero overhead
     * when called from QuantizedGemmKernel hot path.
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
         * @brief Activation-activation GEMM followed by Softmax.
         *
         * Executes C = Softmax(A @ B^T) (axis configurable).
         * Default implementation falls back to false; kernels may override
         * to provide fused implementations via backend-specific acceleration.
         *
         * @param A Left activation matrix [m, k] (FP32)
         * @param B Right activation matrix [n, k] if transpose_B=true (FP32)
         * @param C Output matrix [m, n] containing softmax-normalized scores
         * @param m Number of rows in A and C
         * @param n Number of rows in B (transpose_B=true) or cols (false)
         * @param k Number of columns in A/B when transpose_B=true
         * @param transpose_B Whether to transpose B before multiplication
         * @param softmax_axis Axis over which to apply softmax (0=row-wise, 1=col-wise, -1=last axis)
         * @param mpi_ctx MPI context (unused by most kernels)
         * @param device_idx Device index (-1=CPU)
         *
         * @return true if fused execution succeeded, false otherwise
         */
        virtual bool multiply_activations_with_softmax(
            const float *A, const float *B, float *C,
            int m, int n, int k,
            bool transpose_B = true,
            int softmax_axis = 1,
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
            (void)mpi_ctx;
            (void)device_idx;
            return false;
        }

        /**
         * @brief Weight-tensor GEMM followed by Softmax (same signature as multiply)
         *
         * Executes C = Softmax(A @ B^T) where B is the tensor bound to this kernel.
         * Default implementation returns false so backends can opt-in gradually.
         */
        virtual bool multiply_with_softmax(
            const float *A, float *C,
            int m, int n, int k,
            bool transpose_B = true,
            int softmax_axis = 1,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)A;
            (void)C;
            (void)m;
            (void)n;
            (void)k;
            (void)transpose_B;
            (void)softmax_axis;
            (void)mpi_ctx;
            (void)device_idx;
            return false;
        }
    };

    /**
     * @brief Rotary position embeddings (RoPE)
     */
    class ITensorRoPE : public ITensorKernel
    {
    public:
        /**
         * @brief Apply rotary position embeddings to Q and K (FP32)
         *
         * @param Q Query tensor [seq_len, n_heads, head_dim] (modified in-place)
         * @param K Key tensor [seq_len, n_kv_heads, head_dim] (modified in-place)
         * @param position_ids Position indices [seq_len] (int32)
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads (may be < n_heads for GQA)
         * @param head_dim Dimension per head
         * @param rope_theta RoPE frequency base (10000.0 for LLaMA, 1000000.0 for Qwen2.5)
         * @param use_bf16 Convert to BF16 internally for computation (faster, negligible loss)
         * @param mpi_ctx MPI context (nullptr = single node)
         * @param device_idx Device index for execution (-1 = CPU, ≥0 = GPU)
         *
         * @return true on success, false on error
         */
        virtual bool apply(
            float *Q, float *K,
            const int *position_ids,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            float rope_theta = 10000.0f,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;

        /**
         * @brief Apply RoPE with BF16 tensors (native precision, no conversion)
         *
         * @param Q_bf16 Query tensor in BF16 format [seq_len, n_heads, head_dim]
         * @param K_bf16 Key tensor in BF16 format [seq_len, n_kv_heads, head_dim]
         * @param position_ids Position indices [seq_len]
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads
         * @param head_dim Dimension per head
         * @param rope_theta RoPE frequency base
         * @param device_idx Device index
         *
         * @return true on success, false on error
         *
         * @note Default implementation returns false (opt-in support)
         */
        virtual bool apply_bf16(
            uint16_t *Q_bf16, uint16_t *K_bf16,
            const int *position_ids,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            float rope_theta = 10000.0f,
            int device_idx = -1)
        {
            return false; // Default: not supported
        }

        /**
         * @brief Apply RoPE with FP16 tensors (native precision, no conversion)
         *
         * @param Q_fp16 Query tensor in FP16 format [seq_len, n_heads, head_dim]
         * @param K_fp16 Key tensor in FP16 format [seq_len, n_kv_heads, head_dim]
         * @param position_ids Position indices [seq_len]
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads
         * @param head_dim Dimension per head
         * @param rope_theta RoPE frequency base
         * @param device_idx Device index
         *
         * @return true on success, false on error
         *
         * @note Default implementation returns false (opt-in support)
         */
        virtual bool apply_fp16(
            uint16_t *Q_fp16, uint16_t *K_fp16,
            const int *position_ids,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            float rope_theta = 10000.0f,
            int device_idx = -1)
        {
            return false; // Default: not supported
        }
    };

    /**
     * @brief SwiGLU activation: output = swish(gate) * up
     */
    class ITensorSwiGLU : public ITensorKernel
    {
    public:
        /**
         * @brief Apply SwiGLU activation
         *
         * @param gate Gate projection [seq_len, d_ff]
         * @param up Up projection [seq_len, d_ff]
         * @param output Output [seq_len, d_ff]
         * @param seq_len Sequence length
         * @param d_ff Feed-forward dimension
         * @param use_bf16 Use BF16 for computation (bandwidth-bound operation)
         * @param mpi_ctx MPI context
         * @param device_idx Device index
         *
         * @return true on success
         */
        virtual bool apply(
            const float *gate, const float *up, float *output,
            int seq_len, int d_ff,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;
    };

    /**
     * @brief Softmax normalization
     */
    class ITensorSoftmax : public ITensorKernel
    {
    public:
        /**
         * @brief Apply softmax along last dimension
         *
         * @param input Input tensor [rows, cols]
         * @param output Output tensor [rows, cols]
         * @param rows Number of rows
         * @param cols Number of columns
         * @param use_bf16 Use BF16 (NOT recommended - precision-critical)
         * @param mpi_ctx MPI context
         * @param device_idx Device index
         *
         * @return true on success
         */
        virtual bool apply(
            const float *input, float *output,
            int rows, int cols,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;
    };

    /**
     * @brief RMS normalization
     */
    class ITensorRMSNorm : public ITensorKernel
    {
    public:
        /**
         * @brief Apply RMS normalization (FP32)
         *
         * @param input Input tensor [seq_len, d_model] (FP32)
         * @param gamma Scale parameter [d_model] (FP32)
         * @param output Output tensor [seq_len, d_model] (FP32)
         * @param seq_len Sequence length
         * @param d_model Model dimension
         * @param eps Epsilon for numerical stability
         * @param use_bf16 Use BF16 (NOT recommended - precision-critical)
         * @param mpi_ctx MPI context
         * @param device_idx Device index
         *
         * @return true on success
         */
        virtual bool apply(
            const float *input, const float *gamma, float *output,
            int seq_len, int d_model,
            float eps = 1e-6f,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;

        /**
         * @brief Apply RMS normalization (BF16 native precision)
         *
         * Performs RMS normalization on BF16 tensors without conversion overhead.
         * Internally converts to FP32 for high-precision accumulation, then converts back.
         *
         * @param input_bf16 Input BF16 tensor [seq_len, d_model] (stored as uint16_t)
         * @param gamma Gamma weights [d_model] (FP32)
         * @param output_bf16 Output BF16 tensor [seq_len, d_model] (stored as uint16_t)
         * @param seq_len Sequence length
         * @param d_model Model dimension
         * @param eps Epsilon for numerical stability
         * @param device_idx Device index (-1 for CPU)
         *
         * @return true on success
         */
        virtual bool apply_bf16(
            const uint16_t *input_bf16,
            const float *gamma,
            uint16_t *output_bf16,
            int seq_len,
            int d_model,
            float eps = 1e-6f,
            int device_idx = -1)
        {
            // Default implementation: not supported
            (void)input_bf16;
            (void)gamma;
            (void)output_bf16;
            (void)seq_len;
            (void)d_model;
            (void)eps;
            (void)device_idx;
            return false;
        }

        /**
         * @brief Apply RMS normalization (FP16 native precision)
         *
         * Performs RMS normalization on FP16 tensors without conversion overhead.
         * Internally converts to FP32 for high-precision accumulation, then converts back.
         *
         * @param input_fp16 Input FP16 tensor [seq_len, d_model] (stored as uint16_t)
         * @param gamma Gamma weights [d_model] (FP32)
         * @param output_fp16 Output FP16 tensor [seq_len, d_model] (stored as uint16_t)
         * @param seq_len Sequence length
         * @param d_model Model dimension
         * @param eps Epsilon for numerical stability
         * @param device_idx Device index (-1 for CPU)
         *
         * @return true on success
         */
        virtual bool apply_fp16(
            const uint16_t *input_fp16,
            const float *gamma,
            uint16_t *output_fp16,
            int seq_len,
            int d_model,
            float eps = 1e-6f,
            int device_idx = -1)
        {
            // Default implementation: not supported
            (void)input_fp16;
            (void)gamma;
            (void)output_fp16;
            (void)seq_len;
            (void)d_model;
            (void)eps;
            (void)device_idx;
            return false;
        }

        /**
         * @brief Apply RMS normalization with INT32→INT8 requantization
         *
         * Performs RMS normalization on INT32 accumulator tensors and
         * requantizes output to INT8 with per-row dynamic scaling.
         *
         * Pipeline: INT32 input → normalize → apply gamma → requantize → INT8 output
         *
         * @param input_int32 Input INT32 tensor [seq_len, d_model]
         * @param gamma Scale parameter [d_model] (FP32)
         * @param output_int8 Output INT8 tensor [seq_len, d_model]
         * @param output_row_scales Per-row INT8 scales [seq_len]
         * @param seq_len Sequence length
         * @param d_model Model dimension
         * @param eps Epsilon for numerical stability
         * @param device_idx Device index (-1 for CPU)
         *
         * @return true on success
         */
        virtual bool apply_int32_to_int8(
            const int32_t *input_int32,
            const float *gamma,
            int8_t *output_int8,
            float *output_row_scales,
            int seq_len,
            int d_model,
            float eps = 1e-6f,
            int device_idx = -1)
        {
            // Default implementation: not supported
            (void)input_int32;
            (void)gamma;
            (void)output_int8;
            (void)output_row_scales;
            (void)seq_len;
            (void)d_model;
            (void)eps;
            (void)device_idx;
            return false;
        }
    };

    /**
     * @brief Forward-declared tensor base class
     */
    class TensorBase;

    /**
     * @brief Attention computation kernel (multi-head, grouped-query, multi-query)
     *
     * Implementations:
     * - CPUAttention: OpenMP multi-threaded CPU implementation (from GQAAttention)
     * - CUDAAttention: Flash Attention 2 fused kernel (future)
     * - ROCmAttention: Composable Kernel attention (future)
     *
     * Supports:
     * - Multi-Head Attention (MHA): n_heads == n_kv_heads
     * - Grouped-Query Attention (GQA): n_heads > n_kv_heads, n_heads % n_kv_heads == 0
     * - Multi-Query Attention (MQA): n_kv_heads == 1
     */
    class ITensorAttention : public ITensorKernel
    {
    public:
        /**
         * @brief Compute attention: output = softmax(Q·K^T / sqrt(head_dim)) · V
         *
         * Orchestrates multi-stage attention computation:
         * 1. Broadcast K/V (GQA/MQA only): Expand kv_heads to match n_heads
         * 2. Score computation: scores = Q · K^T
         * 3. Scaling: scores /= sqrt(head_dim)
         * 4. Masking: Apply causal and/or padding masks
         * 5. Softmax: weights = softmax(scores)
         * 6. Context: output = weights · V
         *
         * @param Q Query tensor [seq_len, n_heads, head_dim]
         * @param K Key tensor [seq_len, n_kv_heads, head_dim]
         * @param V Value tensor [seq_len, n_kv_heads, head_dim]
         * @param output Output tensor [seq_len, n_heads, head_dim] (pre-allocated)
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads (≤ n_heads)
         * @param head_dim Dimension per head
         * @param causal Whether to apply causal masking (tokens can only attend to past)
         * @param window_size Sliding window attention (-1 = disabled, >0 = window size)
         * @param workspace_scores Workspace for attention scores [n_heads*seq_len, seq_len] (optional)
         * @param workspace_buffer Workspace for Q/K/V broadcast [num_threads * seq_len * head_dim * 3] (optional)
         * @param workspace_context Workspace for context [num_threads * seq_len * head_dim] (optional)
         * @param workspace_mask Workspace for mask [seq_len, seq_len] (optional)
         * @param use_bf16 Use BF16 for computation (NOT recommended - precision-critical)
         * @param mpi_ctx MPI context
         * @param device_idx Device index
         *
         * @return true on success, false on error
         *
         * @note Workspace tensors are optional hints. If nullptr, kernel allocates internally.
         *       Providing workspaces reduces allocation overhead for repeated calls.
         *
         * @note For GQA/MQA: K/V are broadcast to match n_heads before score computation.
         *       workspace_buffer must be large enough for expanded K/V.
         */
        virtual bool compute(
            const float *Q, const float *K, const float *V,
            float *output,
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
         * @brief Batch attention computation (future extension)
         *
         * Compute attention for multiple independent sequences in parallel.
         * Useful for batched inference workloads.
         *
         * @param Q Query [batch_size, seq_len, n_heads, head_dim]
         * @param K Key [batch_size, seq_len, n_kv_heads, head_dim]
         * @param V Value [batch_size, seq_len, n_kv_heads, head_dim]
         * @param output Output [batch_size, seq_len, n_heads, head_dim]
         * @param batch_size Number of sequences
         * @param seq_len Sequence length (must be same for all sequences)
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads
         * @param head_dim Dimension per head
         * @param causal Whether to apply causal masking
         * @param window_size Sliding window attention (-1 = disabled)
         * @param workspace_scores Workspace [batch * n_heads * seq_len, seq_len]
         * @param workspace_buffer Workspace [batch * num_threads * seq_len * head_dim * 3]
         * @param workspace_context Workspace [batch * num_threads * seq_len * head_dim]
         * @param workspace_mask Workspace [batch * seq_len, seq_len]
         * @param use_bf16 Use BF16 for computation
         * @param mpi_ctx MPI context
         * @param device_idx Device index
         *
         * @return true on success
         */
        virtual bool compute_batch(
            const float *Q, const float *K, const float *V,
            float *output,
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
    };

} // namespace llaminar2
