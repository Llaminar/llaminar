/**
 * @file ROCmFlashAttentionKernelT.h
 * @brief ROCm/HIP Flash Attention kernel implementing ITensorAttention
 *
 * Implements Flash Attention 2 for prefill (seq_len > 1) and
 * Flash Decoding for decode (seq_len = 1).
 *
 * Target Architecture: AMD MI50 (gfx906 / Vega 20)
 *
 * MI50 Architecture Characteristics:
 *   - 64 Compute Units (CUs), 64 wavefronts max per CU
 *   - Wavefront size: 64 threads (vs CUDA's 32)
 *   - 64KB LDS per CU (shared memory equivalent)
 *   - 16GB HBM2 @ 1024 GB/s bandwidth
 *   - No Matrix Cores (added in CDNA1/MI100) - use VALU for GEMM
 *   - FP32 peak: 13.4 TFLOPS, FP16 peak: 26.8 TFLOPS (packed math)
 *   - Supports FP16/BF16 packed math via v_pk_* instructions
 *
 * Algorithm selection is automatic based on seq_len:
 * - seq_len > 1: Flash Attention 2 (tile over Q and K/V with online softmax)
 * - seq_len = 1: Flash Decoding (split-K parallelism over KV cache)
 *
 * Key Differences from CUDA Implementation:
 *   1. No WMMA/Tensor Cores - use vectorized VALU for GEMM
 *   2. Wavefront size 64 vs warp size 32 - adjust reduction patterns
 *   3. LDS (Local Data Share) vs shared memory - same concept, different API
 *   4. No cp.async - use explicit async copy via buffer_load_* intrinsics
 *   5. Different occupancy model - balance LDS/VGPR usage per CU
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../../backends/IWorkerGPUContext.h"
#include "../../../execution/config/RuntimeConfig.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/MPIContext.h"

namespace llaminar2
{
    // Forward declarations for IWorkspaceConsumer
    class DeviceWorkspaceManager;
    struct WorkspaceRequirements;

    namespace rocm
    {
        // =============================================================================
        // Attention Workspace Buffer Names
        // =============================================================================

        /**
         * Standard buffer names for Flash Attention workspace.
         * Flash Decoding requires partial output buffers for split-K reduction.
         */
        namespace AttentionWorkspaceBuffers
        {
            /// Partial attention output [batch × n_heads × num_splits × head_dim] FP32
            constexpr const char *PARTIAL_OUTPUT = "attn_partial_output";
            /// Max scores per split [batch × n_heads × num_splits] FP32
            constexpr const char *PARTIAL_M = "attn_partial_m";
            /// Logsumexp per split [batch × n_heads × num_splits] FP32
            constexpr const char *PARTIAL_L = "attn_partial_l";
        }

        // Forward declaration of precision element type mapping
        namespace detail
        {
            template <ActivationPrecision P>
            struct PrecisionElement;

            template <>
            struct PrecisionElement<ActivationPrecision::FP32>
            {
                using Type = float;
            };

            template <>
            struct PrecisionElement<ActivationPrecision::FP16>
            {
                using Type = uint16_t; // __half via HIP
            };

            template <>
            struct PrecisionElement<ActivationPrecision::BF16>
            {
                using Type = uint16_t; // hip_bfloat16
            };
        } // namespace detail

        /**
         * @brief ROCm Flash Attention kernel template
         *
         * @tparam Precision Activation precision (FP32, FP16, BF16)
         *
         * Implements ITensorAttention interface for AMD GPU execution.
         * Automatically selects between Flash Attention 2 (prefill) and
         * Flash Decoding (decode) based on sequence length.
         *
         * MI50-Specific Optimizations:
         *   - Vectorized FP16 packed math (v_pk_fma_f16) for 2x throughput
         *   - Explicit wavefront-wide reductions using DPP (Data Parallel Primitives)
         *   - LDS bank conflict avoidance with padding
         *   - Occupancy tuning for 64KB LDS per CU
         */
        template <ActivationPrecision Precision>
        class ROCmFlashAttentionKernelT : public ITensorAttention
        {
        public:
            using ElementType = typename detail::PrecisionElement<Precision>::Type;

            /**
             * @brief Construct a ROCm Flash Attention kernel
             * @param device_idx ROCm device index (0-based)
             */
            explicit ROCmFlashAttentionKernelT(int device_idx = 0);
            ~ROCmFlashAttentionKernelT() override;

            // Non-copyable, moveable
            ROCmFlashAttentionKernelT(const ROCmFlashAttentionKernelT &) = delete;
            ROCmFlashAttentionKernelT &operator=(const ROCmFlashAttentionKernelT &) = delete;
            ROCmFlashAttentionKernelT(ROCmFlashAttentionKernelT &&) noexcept;
            ROCmFlashAttentionKernelT &operator=(ROCmFlashAttentionKernelT &&) noexcept;

            /**
             * @brief Check if kernel supports a specific device
             * @param device_idx Device index (-1 = CPU, >=0 = GPU)
             * @return true if device_idx >= 0 (GPU only)
             */
            bool supports_device(int device_idx) const override
            {
                return device_idx >= 0; // GPU only
            }

            void setGPUStream(void *stream) override { stream_ = stream; }

            /**
             * @brief Compute single-sequence attention
             *
             * Dispatches to Flash Attention 2 (seq_len > 1) or Flash Decoding (seq_len = 1).
             *
             * @param Q Query tensor [seq_len, n_heads, head_dim]
             * @param K Key tensor [kv_len, n_kv_heads, head_dim]
             * @param V Value tensor [kv_len, n_kv_heads, head_dim]
             * @param output Output tensor [seq_len, n_heads, head_dim]
             * @param seq_len Query sequence length
             * @param n_heads Number of query heads
             * @param n_kv_heads Number of key/value heads (GQA: n_heads % n_kv_heads == 0)
             * @param head_dim Dimension per head
             * @param causal Apply causal (lower-triangular) masking
             * @param window_size Sliding window size (-1 = disabled)
             * @param workspace_scores Unused (Flash Attention doesn't need score workspace)
             * @param workspace_buffer Unused
             * @param workspace_context Unused
             * @param workspace_mask Pre-built attention mask (optional)
             * @param use_bf16 Ignored (use template parameter)
             * @param mpi_ctx MPI context (unused for now)
             * @param device_idx Device index for execution
             * @return true on success
             */
            bool compute(
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
                int device_idx = -1) override;

            /**
             * @brief Compute batched attention
             *
             * @param Q Query tensor [batch_size * seq_len, n_heads, head_dim]
             * @param K Key tensor [batch_size * kv_len, n_kv_heads, head_dim]
             * @param V Value tensor [batch_size * kv_len, n_kv_heads, head_dim]
             * @param output Output tensor [batch_size * seq_len, n_heads, head_dim]
             * @param batch_size Number of sequences in batch
             * @param seq_len Sequence length per batch item
             * @param n_heads Number of query heads
             * @param n_kv_heads Number of key/value heads
             * @param head_dim Dimension per head
             * @param causal Apply causal masking
             * @param window_size Sliding window size (-1 = disabled)
             * @return true on success
             */
            bool compute_batch(
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
                int device_idx = -1) override;

            /**
             * @brief Compute attention with separate Q and KV lengths (decode mode)
             *
             * Optimized for single-token decode with large KV cache.
             * Uses Flash Decoding algorithm with split-K parallelism.
             *
             * @param Q Query tensor [seq_len, n_heads, head_dim]
             * @param K Key tensor [kv_len, n_kv_heads, head_dim]
             * @param V Value tensor [kv_len, n_kv_heads, head_dim]
             * @param output Output tensor [seq_len, n_heads, head_dim]
             * @param seq_len Query sequence length (typically 1 for decode)
             * @param kv_len KV cache length
             * @param n_heads Number of query heads
             * @param n_kv_heads Number of KV heads
             * @param head_dim Dimension per head
             * @param causal Apply causal masking
             * @param position_offset Starting position (for causal mask)
             * @param device_idx Target device index (-1 uses default)
             * @return true on success
             */
            bool compute_decode(
                const float *Q, const float *K, const float *V, float *output,
                int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
                bool causal = true,
                int position_offset = 0,
                int device_idx = -1);

            /**
             * @brief Internal typed implementation
             *
             * Called by compute() after casting float* to ElementType*.
             */
            bool apply_typed(
                const ElementType *Q, const ElementType *K, const ElementType *V,
                ElementType *output,
                int batch_size, int seq_len, int kv_len,
                int n_heads, int n_kv_heads, int head_dim,
                bool causal, int window_size, int position_offset,
                int device_idx);

            /**
             * @brief Tensor-based attention dispatch
             *
             * Extracts GPU pointers from ITensor and dispatches to compute() or compute_decode().
             * This is the primary entry point for AttentionComputeStage using ITensor* API.
             */
            bool compute_tensor(
                const ITensor *Q,
                const ITensor *K,
                const ITensor *V,
                ITensor *output,
                int batch_size,
                int seq_len,
                int kv_len,
                int n_heads,
                int n_kv_heads,
                int head_dim,
                bool causal = false,
                int window_size = -1,
                ITensor *workspace_scores = nullptr,
                ITensor *workspace_mask = nullptr,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                int head_start = 0,
                int local_n_heads = -1,
                int local_n_kv_heads = -1) override;

        private:
            int device_idx_;
            void *stream_ = nullptr; // hipStream_t

            // Workspace for Flash Decoding partial outputs
            void *partial_output_buf_ = nullptr;
            void *partial_m_buf_ = nullptr; // max scores per split
            void *partial_l_buf_ = nullptr; // logsumexp per split
            size_t workspace_size_ = 0;
            int max_splits_ = 0;

            void allocateWorkspace(int n_heads, int head_dim, int num_splits);
            void freeWorkspace();
        };

        // =====================================================================
        // Full Template Specializations (required for separate compilation)
        // =====================================================================

        /**
         * @brief FP32 Flash Attention kernel specialization
         */
        template <>
        class ROCmFlashAttentionKernelT<ActivationPrecision::FP32> : public ITensorAttention, public IWorkspaceConsumer
        {
        public:
            using ElementType = float;

            /**
             * @brief Create ROCm Flash Attention kernel (legacy constructor)
             * @param device_idx ROCm device index (0-based)
             */
            explicit ROCmFlashAttentionKernelT(int device_idx = 0);

            /**
             * @brief Create ROCm Flash Attention kernel using device context
             *
             * Uses the device context's stream for kernel execution.
             *
             * @param ctx Device context (must outlive this kernel)
             * @throws std::runtime_error if ctx is null or not initialized
             */
            explicit ROCmFlashAttentionKernelT(IWorkerGPUContext *ctx);

            ~ROCmFlashAttentionKernelT() override;

            ROCmFlashAttentionKernelT(const ROCmFlashAttentionKernelT &) = delete;
            ROCmFlashAttentionKernelT &operator=(const ROCmFlashAttentionKernelT &) = delete;
            ROCmFlashAttentionKernelT(ROCmFlashAttentionKernelT &&) noexcept;
            ROCmFlashAttentionKernelT &operator=(ROCmFlashAttentionKernelT &&) noexcept;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            void setGPUStream(void *stream) override { stream_ = stream; }

            bool compute(
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
                int device_idx = -1) override;

            bool compute_batch(
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
                int device_idx = -1) override;

            bool compute_decode(
                const float *Q, const float *K, const float *V, float *output,
                int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
                bool causal = true,
                int position_offset = 0,
                int device_idx = -1);

            bool apply_typed(
                const float *Q, const float *K, const float *V, float *output,
                int batch_size, int seq_len, int kv_len,
                int n_heads, int n_kv_heads, int head_dim,
                bool causal, int window_size, int position_offset,
                int device_idx);

            bool compute_tensor(
                const ITensor *Q,
                const ITensor *K,
                const ITensor *V,
                ITensor *output,
                int batch_size,
                int seq_len,
                int kv_len,
                int n_heads,
                int n_kv_heads,
                int head_dim,
                bool causal = false,
                int window_size = -1,
                ITensor *workspace_scores = nullptr,
                ITensor *workspace_mask = nullptr,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                int head_start = 0,
                int local_n_heads = -1,
                int local_n_kv_heads = -1) override;

            // =========================================================================
            // IWorkspaceConsumer Interface
            // =========================================================================

            /**
             * @brief Get workspace requirements for Flash Decoding
             *
             * Flash Decoding requires partial output buffers for split-K reduction:
             * - partial_output [batch × n_heads × num_splits × head_dim] FP32
             * - partial_m [batch × n_heads × num_splits] FP32 (max scores)
             * - partial_l [batch × n_heads × num_splits] FP32 (logsumexp)
             *
             * @param m Batch size (typically 1 for decode)
             * @param n Number of heads (n_heads)
             * @param k Head dimension (head_dim)
             * @return WorkspaceRequirements with buffer specifications
             */
            WorkspaceRequirements getWorkspaceRequirements(
                int m = 1, int n = 0, int k = 0) const override;

            /**
             * @brief Bind workspace manager for managed mode
             *
             * When bound, Flash Decoding uses pre-allocated buffers instead of
             * internal hipMalloc allocations (which are extremely slow on ROCm).
             *
             * @param workspace Pointer to workspace manager (NOT owned, must outlive kernel)
             */
            void bindWorkspace(DeviceWorkspaceManager *workspace) override;

            /**
             * @brief Check if workspace is currently bound
             */
            bool hasWorkspace() const override;

            /**
             * @brief Get the currently bound workspace manager
             */
            DeviceWorkspaceManager *getWorkspace() const override;

            // =========================================================================
            // Device Context Support (Phase 4)
            // =========================================================================

            /**
             * @brief Set the device context for this kernel
             * @param ctx Device context (owned by GPUDeviceContextPool, not this kernel)
             */
            void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }

            /**
             * @brief Get the currently bound device context
             * @return Device context, or nullptr if not bound
             */
            IWorkerGPUContext *deviceContext() const { return device_ctx_; }

            /**
             * @brief Check if a device context is bound
             * @return true if setDeviceContext() was called with a non-null context
             */
            bool hasDeviceContext() const { return device_ctx_ != nullptr; }

        private:
            int device_idx_;
            void *stream_ = nullptr;
            void *partial_output_buf_ = nullptr;
            void *partial_m_buf_ = nullptr;
            void *partial_l_buf_ = nullptr;
            size_t workspace_size_ = 0;
            int max_splits_ = 0;

            // IWorkspaceConsumer state
            DeviceWorkspaceManager *workspace_ = nullptr;

            // Device Context (Phase 4)
            IWorkerGPUContext *device_ctx_ = nullptr;

            void allocateWorkspace(int n_heads, int head_dim, int num_splits);
            void freeWorkspace();
        };

        /**
         * @brief FP16 Flash Attention kernel specialization
         */
        template <>
        class ROCmFlashAttentionKernelT<ActivationPrecision::FP16> : public ITensorAttention, public IWorkspaceConsumer
        {
        public:
            using ElementType = uint16_t;

            /**
             * @brief Create ROCm Flash Attention kernel (legacy constructor)
             * @param device_idx ROCm device index (0-based)
             */
            explicit ROCmFlashAttentionKernelT(int device_idx = 0);

            /**
             * @brief Create ROCm Flash Attention kernel using device context
             * @param ctx Device context (must outlive this kernel)
             * @throws std::runtime_error if ctx is null or not initialized
             */
            explicit ROCmFlashAttentionKernelT(IWorkerGPUContext *ctx);

            ~ROCmFlashAttentionKernelT() override;

            ROCmFlashAttentionKernelT(const ROCmFlashAttentionKernelT &) = delete;
            ROCmFlashAttentionKernelT &operator=(const ROCmFlashAttentionKernelT &) = delete;
            ROCmFlashAttentionKernelT(ROCmFlashAttentionKernelT &&) noexcept;
            ROCmFlashAttentionKernelT &operator=(ROCmFlashAttentionKernelT &&) noexcept;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            void setGPUStream(void *stream) override { stream_ = stream; }

            bool compute(
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
                int device_idx = -1) override;

            bool compute_batch(
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
                int device_idx = -1) override;

            bool compute_decode(
                const float *Q, const float *K, const float *V, float *output,
                int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
                bool causal = true,
                int position_offset = 0,
                int device_idx = -1);

            bool apply_typed(
                const uint16_t *Q, const uint16_t *K, const uint16_t *V, uint16_t *output,
                int batch_size, int seq_len, int kv_len,
                int n_heads, int n_kv_heads, int head_dim,
                bool causal, int window_size, int position_offset,
                int device_idx);

            bool compute_tensor(
                const ITensor *Q,
                const ITensor *K,
                const ITensor *V,
                ITensor *output,
                int batch_size,
                int seq_len,
                int kv_len,
                int n_heads,
                int n_kv_heads,
                int head_dim,
                bool causal = false,
                int window_size = -1,
                ITensor *workspace_scores = nullptr,
                ITensor *workspace_mask = nullptr,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                int head_start = 0,
                int local_n_heads = -1,
                int local_n_kv_heads = -1) override;

            // =========================================================================
            // IWorkspaceConsumer Interface
            // =========================================================================

            WorkspaceRequirements getWorkspaceRequirements(
                int m = 1, int n = 0, int k = 0) const override;
            void bindWorkspace(DeviceWorkspaceManager *workspace) override;
            bool hasWorkspace() const override;
            DeviceWorkspaceManager *getWorkspace() const override;

            // =========================================================================
            // Device Context Support (Phase 4)
            // =========================================================================

            void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }
            IWorkerGPUContext *deviceContext() const { return device_ctx_; }
            bool hasDeviceContext() const { return device_ctx_ != nullptr; }

        private:
            int device_idx_;
            void *stream_ = nullptr;
            void *partial_output_buf_ = nullptr;
            void *partial_m_buf_ = nullptr;
            void *partial_l_buf_ = nullptr;
            size_t workspace_size_ = 0;
            int max_splits_ = 0;

            // IWorkspaceConsumer state
            DeviceWorkspaceManager *workspace_ = nullptr;

            // Device Context (Phase 4)
            IWorkerGPUContext *device_ctx_ = nullptr;

            void allocateWorkspace(int n_heads, int head_dim, int num_splits);
            void freeWorkspace();
        };

        /**
         * @brief BF16 Flash Attention kernel specialization
         *
         * Note: MI50 has limited BF16 support compared to CDNA. This implementation
         * may need to fall back to FP16 or FP32 accumulation paths.
         */
        template <>
        class ROCmFlashAttentionKernelT<ActivationPrecision::BF16> : public ITensorAttention, public IWorkspaceConsumer
        {
        public:
            using ElementType = uint16_t;

            /**
             * @brief Create ROCm Flash Attention kernel (legacy constructor)
             * @param device_idx ROCm device index (0-based)
             */
            explicit ROCmFlashAttentionKernelT(int device_idx = 0);

            /**
             * @brief Create ROCm Flash Attention kernel using device context
             * @param ctx Device context (must outlive this kernel)
             * @throws std::runtime_error if ctx is null or not initialized
             */
            explicit ROCmFlashAttentionKernelT(IWorkerGPUContext *ctx);

            ~ROCmFlashAttentionKernelT() override;

            ROCmFlashAttentionKernelT(const ROCmFlashAttentionKernelT &) = delete;
            ROCmFlashAttentionKernelT &operator=(const ROCmFlashAttentionKernelT &) = delete;
            ROCmFlashAttentionKernelT(ROCmFlashAttentionKernelT &&) noexcept;
            ROCmFlashAttentionKernelT &operator=(ROCmFlashAttentionKernelT &&) noexcept;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            void setGPUStream(void *stream) override { stream_ = stream; }

            bool compute(
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
                int device_idx = -1) override;

            bool compute_batch(
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
                int device_idx = -1) override;

            bool compute_decode(
                const float *Q, const float *K, const float *V, float *output,
                int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
                bool causal = true,
                int position_offset = 0,
                int device_idx = -1);

            bool apply_typed(
                const uint16_t *Q, const uint16_t *K, const uint16_t *V, uint16_t *output,
                int batch_size, int seq_len, int kv_len,
                int n_heads, int n_kv_heads, int head_dim,
                bool causal, int window_size, int position_offset,
                int device_idx);

            bool compute_tensor(
                const ITensor *Q,
                const ITensor *K,
                const ITensor *V,
                ITensor *output,
                int batch_size,
                int seq_len,
                int kv_len,
                int n_heads,
                int n_kv_heads,
                int head_dim,
                bool causal = false,
                int window_size = -1,
                ITensor *workspace_scores = nullptr,
                ITensor *workspace_mask = nullptr,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                int head_start = 0,
                int local_n_heads = -1,
                int local_n_kv_heads = -1) override;

            // =========================================================================
            // IWorkspaceConsumer Interface
            // =========================================================================

            WorkspaceRequirements getWorkspaceRequirements(
                int m = 1, int n = 0, int k = 0) const override;
            void bindWorkspace(DeviceWorkspaceManager *workspace) override;
            bool hasWorkspace() const override;
            DeviceWorkspaceManager *getWorkspace() const override;

            // =========================================================================
            // Device Context Support (Phase 4)
            // =========================================================================

            void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }
            IWorkerGPUContext *deviceContext() const { return device_ctx_; }
            bool hasDeviceContext() const { return device_ctx_ != nullptr; }

        private:
            int device_idx_;
            void *stream_ = nullptr;
            void *partial_output_buf_ = nullptr;
            void *partial_m_buf_ = nullptr;
            void *partial_l_buf_ = nullptr;
            size_t workspace_size_ = 0;
            int max_splits_ = 0;

            // IWorkspaceConsumer state
            DeviceWorkspaceManager *workspace_ = nullptr;

            // Device Context (Phase 4)
            IWorkerGPUContext *device_ctx_ = nullptr;

            void allocateWorkspace(int n_heads, int head_dim, int num_splits);
            void freeWorkspace();
        };

        // Type aliases for convenience
        using ROCmFlashAttentionKernelFP32 = ROCmFlashAttentionKernelT<ActivationPrecision::FP32>;
        using ROCmFlashAttentionKernelFP16 = ROCmFlashAttentionKernelT<ActivationPrecision::FP16>;
        using ROCmFlashAttentionKernelBF16 = ROCmFlashAttentionKernelT<ActivationPrecision::BF16>;

    } // namespace rocm
} // namespace llaminar2
