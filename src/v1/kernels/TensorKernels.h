#pragma once

#include <memory>
#include <string>
#include "MPIContext.h"
#include "ComputeBackend.h"

namespace llaminar
{

/**
 * @brief Base interface for tensor-attached fused kernels
 * 
 * All kernels accept both MPIContext (for distributed coordination) and
 * ComputeContext (for device selection). This enables:
 * - CPU inference (single node)
 * - CPU inference (multi-node MPI)
 * - GPU inference (single GPU)
 * - GPU inference (multi-GPU, single node via NCCL/RCCL)
 * - GPU inference (multi-GPU, multi-node via MPI + NCCL/RCCL)
 */
class ITensorKernel
{
public:
    virtual ~ITensorKernel() = default;
    virtual const char* name() const = 0;
    
    /**
     * @brief Check if kernel supports given compute backend
     */
    virtual bool supports_backend(ComputeBackendType backend) const = 0;
};

/**
 * @brief GEMM kernel interface with fused streaming dequantization
 * 
 * Supports:
 * - FP32×FP32 → FP32 (baseline)
 * - BF16×BF16 → FP32 (2× bandwidth reduction)
 * - IQ4_NL×FP32 → FP32 (streaming dequant, 8× compression)
 * - INT8×INT8 → INT32 → FP32 (future quantized inference)
 */
class ITensorGemm : public ITensorKernel
{
public:
    /**
     * @brief Matrix multiplication: C = A @ B
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
        const float* A, float* C,
        int m, int n, int k,
        bool transpose_B = true,
        float alpha = 1.0f, float beta = 0.0f,
        const MPIContext* mpi_ctx = nullptr,
        int device_idx = -1) = 0;

    /**
     * @brief Check if kernel supports given operation size
     * 
     * Some kernels may have size limits or optimality ranges.
     */
    virtual bool supports(int m, int n, int k) const = 0;
};

/**
 * @brief RoPE (Rotary Position Embedding) kernel with optional BF16 computation
 * 
 * Applies rotary position encoding to Q and K tensors:
 * - Precomputed sin/cos tables for efficiency
 * - Optional BF16 computation (2× bandwidth reduction, minimal precision loss)
 * - GPU acceleration via CUDA/ROCm/Vulkan compute shaders
 */
class ITensorRoPE : public ITensorKernel
{
public:
    /**
     * @brief Apply rotary position embeddings to Q and K
     * 
     * @param Q Query tensor [seq_len, n_heads, head_dim] (modified in-place)
     * @param K Key tensor [seq_len, n_kv_heads, head_dim] (modified in-place)
     * @param position_ids Position indices [seq_len] (int32)
     * @param seq_len Sequence length
     * @param n_heads Number of query heads
     * @param n_kv_heads Number of key/value heads (may be < n_heads for GQA)
     * @param head_dim Dimension per head
     * @param use_bf16 Convert to BF16 internally for computation (faster, negligible loss)
     * @param mpi_ctx MPI context (nullptr = single node)
     * @param device_idx Device index for execution (-1 = CPU, ≥0 = GPU)
     * 
     * @return true on success, false on error
     */
    virtual bool apply(
        float* Q, float* K,
        const int* position_ids,
        int seq_len, int n_heads, int n_kv_heads, int head_dim,
        bool use_bf16 = false,
        const MPIContext* mpi_ctx = nullptr,
        int device_idx = -1) = 0;
};

/**
 * @brief SwiGLU activation kernel: out = gate ⊙ SiLU(up)
 * 
 * Fused gated activation used in modern transformers (LLaMA, Qwen, etc.)
 * SiLU(x) = x / (1 + exp(-x))
 */
class ITensorSwiGLU : public ITensorKernel
{
public:
    /**
     * @brief Apply SwiGLU activation
     * 
     * @param gate Gate projection tensor [seq_len, d_ff] (input)
     * @param up Up projection tensor [seq_len, d_ff] (input)
     * @param output Output tensor [seq_len, d_ff] (output)
     * @param seq_len Sequence length
     * @param d_ff FFN intermediate dimension
     * @param use_bf16 Use BF16 internally (faster element-wise ops)
     * @param mpi_ctx MPI context (nullptr = single node)
     * @param compute_ctx Compute context for device selection
     * 
     * @return true on success, false on error
     */
    virtual bool apply(
        const float* gate, const float* up, float* output,
        int seq_len, int d_ff,
        bool use_bf16 = false,
        const MPIContext* mpi_ctx = nullptr,
        int device_idx = -1) = 0;
};

/**
 * @brief Softmax kernel with numerical stability
 * 
 * Numerically stable softmax: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 * Critical for attention mechanism correctness.
 */
class ITensorSoftmax : public ITensorKernel
{
public:
    /**
     * @brief Apply softmax along last dimension
     * 
     * @param input Input tensor [batch, seq_len, seq_len] or [seq_len, seq_len]
     * @param output Output tensor (same shape as input)
     * @param rows Number of rows (batch * seq_len)
     * @param cols Number of columns (seq_len)
     * @param use_bf16 Use BF16 (UNSAFE - may lose precision in exp/sum)
     * @param mpi_ctx MPI context (nullptr = single node)
     * @param compute_ctx Compute context for device selection
     * 
     * @return true on success, false on error
     * 
     * @note use_bf16=false recommended for numerical stability
     */
    virtual bool apply(
        const float* input, float* output,
        int rows, int cols,
        bool use_bf16 = false,
        const MPIContext* mpi_ctx = nullptr,
        int device_idx = -1) = 0;
};

/**
 * @brief RMSNorm kernel: x / RMS(x) * gamma
 * 
 * RMS(x) = sqrt(mean(x^2) + eps)
 * Critical for training stability, requires FP32 for variance computation.
 */
class ITensorRMSNorm : public ITensorKernel
{
public:
    /**
     * @brief Apply RMSNorm
     * 
     * @param input Input tensor [seq_len, d_model]
     * @param gamma Scale parameter [d_model]
     * @param output Output tensor [seq_len, d_model]
     * @param seq_len Sequence length
     * @param d_model Model dimension
     * @param eps Epsilon for numerical stability (default: 1e-6)
     * @param use_bf16 Use BF16 (UNSAFE - may lose precision in variance)
     * @param mpi_ctx MPI context (nullptr = single node)
     * @param compute_ctx Compute context for device selection
     * 
     * @return true on success, false on error
     * 
     * @note use_bf16=false recommended for numerical stability
     */
    virtual bool apply(
        const float* input, const float* gamma, float* output,
        int seq_len, int d_model,
        float eps = 1e-6f,
        bool use_bf16 = false,
        const MPIContext* mpi_ctx = nullptr,
        int device_idx = -1) = 0;
};

} // namespace llaminar
