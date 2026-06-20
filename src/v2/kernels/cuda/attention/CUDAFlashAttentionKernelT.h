/**
 * @file CUDAFlashAttentionKernelT.h
 * @brief CUDA Flash Attention kernel implementing ITensorAttention
 *
 * Implements Flash Attention 2 for prefill (seq_len > 1) and
 * Flash Decoding for decode (seq_len = 1).
 *
 * Algorithm selection is automatic based on seq_len:
 * - seq_len > 1: Flash Attention 2 (tile over Q and K/V with online softmax)
 * - seq_len = 1: Flash Decoding (split-K parallelism over KV cache)
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
#include "../../attention/AttentionDeviceParams.h"
#include <array>

namespace llaminar2
{
    // Forward declarations for IWorkspaceConsumer
    class DeviceWorkspaceManager;
    struct WorkspaceRequirements;

    namespace cuda
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
            /// Device-resident dynamic params (kv_len, position_offset, mask_stride)
            constexpr const char *DEVICE_PARAMS = "attn_device_params";
            /// Temporary FP32 K buffer for mixed-precision KV conversion
            constexpr const char *K_TMP_FP32 = "attn_k_tmp_fp32";
            /// Temporary FP32 V buffer for mixed-precision KV conversion
            constexpr const char *V_TMP_FP32 = "attn_v_tmp_fp32";
        }

        /**
         * @brief Maximum verifier/decode rows that share one dynamic attention-param upload.
         *
         * The CUDA small-M attention path currently supports verifier groups up to
         * four rows.  Keeping this value in the header lets both the workspace
         * declaration and the host staging storage stay in lockstep.
         */
        constexpr int kMaxDynamicAttentionParamRows = 4;
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
                using Type = uint16_t;
            };

            template <>
            struct PrecisionElement<ActivationPrecision::BF16>
            {
                using Type = uint16_t;
            };
        } // namespace detail

        /**
         * @brief CUDA Flash Attention kernel template
         *
         * @tparam Precision Activation precision (FP32, FP16, BF16)
         *
         * Implements ITensorAttention interface for GPU execution.
         * Automatically selects between Flash Attention 2 (prefill) and
         * Flash Decoding (decode) based on sequence length.
         */
        template <ActivationPrecision Precision>
        class CUDAFlashAttentionKernelT : public ITensorAttention
        {
        public:
            using ElementType = typename detail::PrecisionElement<Precision>::Type;

            /**
             * @brief Construct a CUDA Flash Attention kernel
             * @param device_idx CUDA device index (0-based)
             */
            explicit CUDAFlashAttentionKernelT(int device_idx = -1);
            ~CUDAFlashAttentionKernelT() override;

            // Non-copyable, moveable
            CUDAFlashAttentionKernelT(const CUDAFlashAttentionKernelT &) = delete;
            CUDAFlashAttentionKernelT &operator=(const CUDAFlashAttentionKernelT &) = delete;
            CUDAFlashAttentionKernelT(CUDAFlashAttentionKernelT &&) noexcept;
            CUDAFlashAttentionKernelT &operator=(CUDAFlashAttentionKernelT &&) noexcept;

            /**
             * @brief Check if kernel supports a specific device
             * @param device_idx Device index (-1 = CPU, >=0 = GPU)
             * @return true if device_idx >= 0 (GPU only)
             */
            bool supports_device(int device_idx) const override
            {
                return device_idx >= 0; // GPU only
            }

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
                const IMPIContext *mpi_ctx = nullptr,
                int device_idx = -1);

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
                const IMPIContext *mpi_ctx = nullptr,
                int device_idx = -1);

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
             * @return true on success
             */
            bool compute_decode(
                const float *Q, const float *K, const float *V, float *output,
                int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
                bool causal = true,
                int position_offset = 0);

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
                int device_idx,
                const attention::AttentionDeviceParams *device_params = nullptr,
                const float *mask = nullptr,
                int head_start = 0,
                int gqa_n_rep = 0);

        private:
            int device_idx_;
            void *stream_ = nullptr; // cudaStream_t

            // Workspace for Flash Decoding partial outputs
            void *partial_output_buf_ = nullptr;
            void *partial_m_buf_ = nullptr;
            void *partial_l_buf_ = nullptr;
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
        class CUDAFlashAttentionKernelT<ActivationPrecision::FP32> : public ITensorAttention, public IWorkspaceConsumer
        {
        public:
            using ElementType = float;

            /**
             * @brief Create CUDA Flash Attention kernel (legacy constructor)
             * @param device_idx CUDA device index (0-based)
             */
            explicit CUDAFlashAttentionKernelT(int device_idx = -1);

            /**
             * @brief Create CUDA Flash Attention kernel using device context
             *
             * Uses the device context's stream for kernel execution.
             *
             * @param ctx Device context (must outlive this kernel)
             * @throws std::runtime_error if ctx is null or not initialized
             */
            explicit CUDAFlashAttentionKernelT(IWorkerGPUContext *ctx);

            ~CUDAFlashAttentionKernelT() override;

            CUDAFlashAttentionKernelT(const CUDAFlashAttentionKernelT &) = delete;
            CUDAFlashAttentionKernelT &operator=(const CUDAFlashAttentionKernelT &) = delete;
            CUDAFlashAttentionKernelT(CUDAFlashAttentionKernelT &&) noexcept;
            CUDAFlashAttentionKernelT &operator=(CUDAFlashAttentionKernelT &&) noexcept;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            void setGPUStream(void *stream) override { stream_ = stream; }

            /// Update attention params in pinned host memory for graph replay
            void setDynamicAttnParams(int kv_len, int position_offset) override;
            void setDynamicAttnParams(int kv_len, int position_offset, int query_rows) override;
            bool prepareDynamicAttnParams(
                int kv_len, int position_offset, int query_rows, void *stream) override;
            bool prepareDynamicAttnParamsFromDeviceSequenceState(
                const int *post_append_cached_tokens_device,
                int seq_len,
                int query_rows,
                void *stream) override;

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
                const IMPIContext *mpi_ctx = nullptr,
                int device_idx = -1);

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
                const IMPIContext *mpi_ctx = nullptr,
                int device_idx = -1);

            bool compute_decode(
                const float *Q, const float *K, const float *V, float *output,
                int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
                bool causal = true,
                int position_offset = 0);

            bool apply_typed(
                const float *Q, const float *K, const float *V, float *output,
                int batch_size, int seq_len, int kv_len,
                int n_heads, int n_kv_heads, int head_dim,
                bool causal, int window_size, int position_offset,
                int device_idx,
                const attention::AttentionDeviceParams *device_params = nullptr,
                const float *mask = nullptr,
                int head_start = 0,
                int gqa_n_rep = 0);

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
                const IMPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                int head_start = 0,
                int local_n_heads = -1,
                int local_n_kv_heads = -1,
                int gqa_n_rep = 0) override;

            /**
             * @brief Compute compact MTP verifier rows through the GPU small-M decode path.
             *
             * This is the graph-stage proof boundary for M=2..4 verifier rows.
             * It prepares row-local attention parameters for the whole verifier
             * span, then invokes one small-M attention dispatch.  The current
             * CUDA backend still executes row-local flash-decode launches inside
             * that dispatch, so this is correctness plumbing, not the final
             * hardware-ceiling grouped attention kernel.
             */
            bool compute_verifier_rows_decode_equivalent(
                const ITensor *Q,
                const ITensor *K,
                const ITensor *V,
                ITensor *output,
                int verifier_rows,
                int kv_len,
                int n_heads,
                int n_kv_heads,
                int head_dim,
                bool causal,
                int window_size = -1,
                const IMPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                int head_start = 0,
                int gqa_n_rep = 0) override;

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
             * Flash Decoding requires pre-allocated buffers from this workspace.
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

            /**
             * @brief Fused TQ KV decode attention (TQ8 K + TQ4 V with rotation trick)
             *
             * Called directly by AttentionComputeStage when TQ GPU decode is detected.
             * Bypasses normal compute_tensor() dispatch since TQ requires ring buffer
             * metadata, rotation matrices, and codebook access.
             */
            bool compute_tensor_tq_decode(
                const ITensor *Q,
                ITensor *output,
                const void *K_cache,
                const void *V_cache,
                const float *rotation,
                const float *rotation_t,
                int batch_size,
                int kv_count,
                int n_heads,
                int n_kv_heads,
                int head_dim,
                int max_seq_len,
                int tail,
                int k_block_size,
                int v_block_size,
                int head_start = 0,
                int gqa_n_rep = 0);

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

            /// Fixed host staging for attention params that are uploaded before graph capture.
            std::array<attention::AttentionDeviceParams, kMaxDynamicAttentionParamRows> h_attn_params_{};
            int h_attn_params_capacity_ = kMaxDynamicAttentionParamRows;
            int dynamic_attn_kv_len_ = 0;
            int dynamic_attn_position_offset_ = 0;
            int dynamic_attn_query_rows_ = 1;
            int dynamic_attn_param_rows_ = 1;
            bool dynamic_attn_host_valid_ = false;
            bool dynamic_attn_device_valid_ = false;
            bool dynamic_attn_device_derived_ = false;

            /**
             * @brief Validate that fixed host staging can hold the requested rows.
             *
             * No CUDA allocation is allowed here: callers may prepare attention
             * params during lazy graph setup, and the actual device storage lives
             * in the IWorkspaceConsumer buffer named @ref AttentionWorkspaceBuffers::DEVICE_PARAMS.
             */
            bool ensureHostAttnParamsCapacity(int capacity);
            bool uploadDynamicAttnParams(void *stream);
            bool dynamicAttnParamsReady(
                int kv_len, int position_offset, int query_rows) const;
            bool allocateWorkspace(int n_heads, int head_dim, int num_splits);
            void freeWorkspace();
        };

        /**
         * @brief FP16 Flash Attention kernel specialization
         */
        template <>
        class CUDAFlashAttentionKernelT<ActivationPrecision::FP16> : public ITensorAttention, public IWorkspaceConsumer
        {
        public:
            using ElementType = uint16_t;

            /**
             * @brief Create CUDA Flash Attention kernel (legacy constructor)
             * @param device_idx CUDA device index (0-based)
             */
            explicit CUDAFlashAttentionKernelT(int device_idx = -1);

            /**
             * @brief Create CUDA Flash Attention kernel using device context
             * @param ctx Device context (must outlive this kernel)
             * @throws std::runtime_error if ctx is null or not initialized
             */
            explicit CUDAFlashAttentionKernelT(IWorkerGPUContext *ctx);

            ~CUDAFlashAttentionKernelT() override;

            CUDAFlashAttentionKernelT(const CUDAFlashAttentionKernelT &) = delete;
            CUDAFlashAttentionKernelT &operator=(const CUDAFlashAttentionKernelT &) = delete;
            CUDAFlashAttentionKernelT(CUDAFlashAttentionKernelT &&) noexcept;
            CUDAFlashAttentionKernelT &operator=(CUDAFlashAttentionKernelT &&) noexcept;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            /// Update attention params in pinned host memory for graph replay
            void setDynamicAttnParams(int kv_len, int position_offset) override;

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
                const IMPIContext *mpi_ctx = nullptr,
                int device_idx = -1);

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
                const IMPIContext *mpi_ctx = nullptr,
                int device_idx = -1);

            bool compute_decode(
                const float *Q, const float *K, const float *V, float *output,
                int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
                bool causal = true,
                int position_offset = 0);

            bool apply_typed(
                const uint16_t *Q, const uint16_t *K, const uint16_t *V, uint16_t *output,
                int batch_size, int seq_len, int kv_len,
                int n_heads, int n_kv_heads, int head_dim,
                bool causal, int window_size, int position_offset,
                int device_idx,
                const attention::AttentionDeviceParams *device_params = nullptr,
                const float *mask = nullptr);

            /**
             * @brief Tensor-based attention dispatch for FP16
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
                const IMPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                int head_start = 0,
                int local_n_heads = -1,
                int local_n_kv_heads = -1,
                int gqa_n_rep = 0) override;

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

            /// Single-row host staging for attention params uploaded to DEVICE_PARAMS.
            attention::AttentionDeviceParams h_attn_params_{};
            bool dynamic_attn_device_valid_ = false;

            void allocateWorkspace(int n_heads, int head_dim, int num_splits);
            void freeWorkspace();
        };

        /**
         * @brief BF16 Flash Attention kernel specialization
         */
        template <>
        class CUDAFlashAttentionKernelT<ActivationPrecision::BF16> : public ITensorAttention, public IWorkspaceConsumer
        {
        public:
            using ElementType = uint16_t;

            /**
             * @brief Create CUDA Flash Attention kernel (legacy constructor)
             * @param device_idx CUDA device index (0-based)
             */
            explicit CUDAFlashAttentionKernelT(int device_idx = -1);

            /**
             * @brief Create CUDA Flash Attention kernel using device context
             * @param ctx Device context (must outlive this kernel)
             * @throws std::runtime_error if ctx is null or not initialized
             */
            explicit CUDAFlashAttentionKernelT(IWorkerGPUContext *ctx);

            ~CUDAFlashAttentionKernelT() override;

            CUDAFlashAttentionKernelT(const CUDAFlashAttentionKernelT &) = delete;
            CUDAFlashAttentionKernelT &operator=(const CUDAFlashAttentionKernelT &) = delete;
            CUDAFlashAttentionKernelT(CUDAFlashAttentionKernelT &&) noexcept;
            CUDAFlashAttentionKernelT &operator=(CUDAFlashAttentionKernelT &&) noexcept;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            /// Update attention params in pinned host memory for graph replay
            void setDynamicAttnParams(int kv_len, int position_offset) override;

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
                const IMPIContext *mpi_ctx = nullptr,
                int device_idx = -1);

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
                const IMPIContext *mpi_ctx = nullptr,
                int device_idx = -1);

            bool compute_decode(
                const float *Q, const float *K, const float *V, float *output,
                int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
                bool causal = true,
                int position_offset = 0);

            bool apply_typed(
                const uint16_t *Q, const uint16_t *K, const uint16_t *V, uint16_t *output,
                int batch_size, int seq_len, int kv_len,
                int n_heads, int n_kv_heads, int head_dim,
                bool causal, int window_size, int position_offset,
                int device_idx,
                const attention::AttentionDeviceParams *device_params = nullptr,
                const float *mask = nullptr);

            /**
             * @brief Tensor-based attention dispatch for BF16
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
                const IMPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                int head_start = 0,
                int local_n_heads = -1,
                int local_n_kv_heads = -1,
                int gqa_n_rep = 0) override;

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

            /// Single-row host staging for attention params uploaded to DEVICE_PARAMS.
            attention::AttentionDeviceParams h_attn_params_{};
            bool dynamic_attn_device_valid_ = false;

            void allocateWorkspace(int n_heads, int head_dim, int num_splits);
            void freeWorkspace();
        };

        // Type aliases for convenience
        using CUDAFlashAttentionKernelFP32 = CUDAFlashAttentionKernelT<ActivationPrecision::FP32>;
        using CUDAFlashAttentionKernelFP16 = CUDAFlashAttentionKernelT<ActivationPrecision::FP16>;
        using CUDAFlashAttentionKernelBF16 = CUDAFlashAttentionKernelT<ActivationPrecision::BF16>;

    } // namespace cuda
} // namespace llaminar2
