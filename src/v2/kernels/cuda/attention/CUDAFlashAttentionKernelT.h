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

#include "../../../execution/RuntimeConfig.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/MPIContext.h"

namespace llaminar2
{
    namespace cuda
    {
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
            explicit CUDAFlashAttentionKernelT(int device_idx = 0);
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
                int device_idx);

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
        class CUDAFlashAttentionKernelT<ActivationPrecision::FP32> : public ITensorAttention
        {
        public:
            using ElementType = float;

            explicit CUDAFlashAttentionKernelT(int device_idx = 0);
            ~CUDAFlashAttentionKernelT() override;

            CUDAFlashAttentionKernelT(const CUDAFlashAttentionKernelT &) = delete;
            CUDAFlashAttentionKernelT &operator=(const CUDAFlashAttentionKernelT &) = delete;
            CUDAFlashAttentionKernelT(CUDAFlashAttentionKernelT &&) noexcept;
            CUDAFlashAttentionKernelT &operator=(CUDAFlashAttentionKernelT &&) noexcept;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

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
                int position_offset = 0);

            bool apply_typed(
                const float *Q, const float *K, const float *V, float *output,
                int batch_size, int seq_len, int kv_len,
                int n_heads, int n_kv_heads, int head_dim,
                bool causal, int window_size, int position_offset,
                int device_idx);

            /**
             * @brief Tensor-based attention dispatch
             *
             * Extracts GPU pointers from TensorBase and dispatches to compute() or compute_decode().
             * This is the primary entry point for AttentionComputeStage using ITensor* API.
             */
            bool compute_tensor(
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
                int head_start = 0,
                int local_n_heads = -1,
                int local_n_kv_heads = -1) override;

        private:
            int device_idx_;
            void *stream_ = nullptr;
            void *partial_output_buf_ = nullptr;
            void *partial_m_buf_ = nullptr;
            void *partial_l_buf_ = nullptr;
            size_t workspace_size_ = 0;
            int max_splits_ = 0;

            void allocateWorkspace(int n_heads, int head_dim, int num_splits);
            void freeWorkspace();
        };

        /**
         * @brief FP16 Flash Attention kernel specialization
         */
        template <>
        class CUDAFlashAttentionKernelT<ActivationPrecision::FP16> : public ITensorAttention
        {
        public:
            using ElementType = uint16_t;

            explicit CUDAFlashAttentionKernelT(int device_idx = 0);
            ~CUDAFlashAttentionKernelT() override;

            CUDAFlashAttentionKernelT(const CUDAFlashAttentionKernelT &) = delete;
            CUDAFlashAttentionKernelT &operator=(const CUDAFlashAttentionKernelT &) = delete;
            CUDAFlashAttentionKernelT(CUDAFlashAttentionKernelT &&) noexcept;
            CUDAFlashAttentionKernelT &operator=(CUDAFlashAttentionKernelT &&) noexcept;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

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
                int position_offset = 0);

            bool apply_typed(
                const uint16_t *Q, const uint16_t *K, const uint16_t *V, uint16_t *output,
                int batch_size, int seq_len, int kv_len,
                int n_heads, int n_kv_heads, int head_dim,
                bool causal, int window_size, int position_offset,
                int device_idx);

            /**
             * @brief Tensor-based attention dispatch for FP16
             */
            bool compute_tensor(
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
                int head_start = 0,
                int local_n_heads = -1,
                int local_n_kv_heads = -1) override;

        private:
            int device_idx_;
            void *stream_ = nullptr;
            void *partial_output_buf_ = nullptr;
            void *partial_m_buf_ = nullptr;
            void *partial_l_buf_ = nullptr;
            size_t workspace_size_ = 0;
            int max_splits_ = 0;

            void allocateWorkspace(int n_heads, int head_dim, int num_splits);
            void freeWorkspace();
        };

        /**
         * @brief BF16 Flash Attention kernel specialization
         */
        template <>
        class CUDAFlashAttentionKernelT<ActivationPrecision::BF16> : public ITensorAttention
        {
        public:
            using ElementType = uint16_t;

            explicit CUDAFlashAttentionKernelT(int device_idx = 0);
            ~CUDAFlashAttentionKernelT() override;

            CUDAFlashAttentionKernelT(const CUDAFlashAttentionKernelT &) = delete;
            CUDAFlashAttentionKernelT &operator=(const CUDAFlashAttentionKernelT &) = delete;
            CUDAFlashAttentionKernelT(CUDAFlashAttentionKernelT &&) noexcept;
            CUDAFlashAttentionKernelT &operator=(CUDAFlashAttentionKernelT &&) noexcept;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

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
                int position_offset = 0);

            bool apply_typed(
                const uint16_t *Q, const uint16_t *K, const uint16_t *V, uint16_t *output,
                int batch_size, int seq_len, int kv_len,
                int n_heads, int n_kv_heads, int head_dim,
                bool causal, int window_size, int position_offset,
                int device_idx);

            /**
             * @brief Tensor-based attention dispatch for BF16
             */
            bool compute_tensor(
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
                int head_start = 0,
                int local_n_heads = -1,
                int local_n_kv_heads = -1) override;

        private:
            int device_idx_;
            void *stream_ = nullptr;
            void *partial_output_buf_ = nullptr;
            void *partial_m_buf_ = nullptr;
            void *partial_l_buf_ = nullptr;
            size_t workspace_size_ = 0;
            int max_splits_ = 0;

            void allocateWorkspace(int n_heads, int head_dim, int num_splits);
            void freeWorkspace();
        };

        // Type aliases for convenience
        using CUDAFlashAttentionKernelFP32 = CUDAFlashAttentionKernelT<ActivationPrecision::FP32>;
        using CUDAFlashAttentionKernelFP16 = CUDAFlashAttentionKernelT<ActivationPrecision::FP16>;
        using CUDAFlashAttentionKernelBF16 = CUDAFlashAttentionKernelT<ActivationPrecision::BF16>;

    } // namespace cuda
} // namespace llaminar2
