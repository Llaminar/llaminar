/**
 * @file CPUAttentionT.h
 * @brief Template-based CPU attention kernel supporting all activation precisions
 *
 * Replaces CPUAttention with a precision-agnostic template implementation.
 * Uses ActivationTraits for precision-specific operations (softmax, GEMM, workspace allocation).
 *
 * Benefits:
 * - Single implementation supports FP32, BF16, FP16, INT32 (zero code duplication)
 * - Compile-time dispatch via ActivationTraits (zero runtime overhead)
 * - Type-safe (compile errors for unsupported types)
 * - Eliminates dummy tensor creation (uses traits directly)
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../tensors/TensorKernels.h"
#include "../../tensors/Tensors.h"
#include "../../tensors/SIMDHelpers.h"
#include "primitives/ActivationTraits.h"
#include "primitives/SoftmaxPrimitives_New.h"
#include "../../pipelines/AttentionUtils.h"
#include "../../utils/Logger.h"
#include <memory>
#include <vector>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>
#include <omp.h>

namespace llaminar2
{

    /**
     * @brief Template-based CPU attention kernel
     *
     * @tparam TensorType Tensor type (FP32Tensor, BF16Tensor, FP16Tensor, INT32Tensor)
     *
     * Uses ActivationTraits<TensorType> for precision-specific operations:
     * - apply_softmax(): Native precision softmax (FP32/BF16/FP16) or conversion (INT32)
     * - create_activation_gemm(): Precision-specific GEMM kernel
     * - allocate_workspace(): Precision-specific workspace allocation
     *
     * Thread safety:
     * - Uses per-thread buffers in workspace_buffer and workspace_context
     * - Thread-safe for concurrent calls with different workspaces
     * - NOT thread-safe for concurrent calls sharing workspaces
     *
     * Performance:
     * - Compile-time dispatch (zero runtime overhead vs CPUAttention)
     * - SIMD softmax variants (AVX512/AVX2/scalar) from tested primitives
     * - Strided GEMM for zero-copy multi-head processing
     * - Fused attention scaling in Q@K^T GEMM alpha parameter
     */
    template <typename TensorType>
    class CPUAttentionT : public ITensorAttention
    {
    public:
        // Infer ElementType from ActivationTraits (which knows the mapping)
        using ElementType = typename primitives::ActivationTraits<TensorType>::ElementType;
        using Traits = primitives::ActivationTraits<TensorType>;

        CPUAttentionT() = default;
        ~CPUAttentionT() override = default;

        /**
         * @brief Check if kernel supports specific device
         *
         * @param device_idx Device index (-1 = CPU, ≥0 = GPU)
         * @return true if device_idx == -1 (CPU only)
         */
        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only
        }

        /**
         * @brief Compute attention (ITensorAttention interface implementation)
         *
         * Takes float* parameters (interface requirement) and casts to ElementType* internally.
         * For FP32Tensor: ElementType=float, no cast needed
         * For BF16/FP16/INT32Tensor: ElementType=uint16_t/int32_t, requires cast
         *
         * Algorithm:
         * 1. Broadcast K/V if GQA/MQA (n_heads != n_kv_heads)
         * 2. Per-head Q@K^T with fused scaling (1/sqrt(d_k)) via GEMM alpha
         * 3. Softmax with optional causal masking (native precision via traits)
         * 4. Context = scores@V using strided GEMM (zero-copy multi-head)
         */
        bool compute(
            const float *Q, const float *K, const float *V,
            float *output,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal = false,
            int window_size = -1,
            TensorBase *workspace_scores = nullptr,
            TensorBase *workspace_buffer = nullptr,
            TensorBase *workspace_context = nullptr,
            TensorBase *workspace_mask = nullptr,
            bool use_bf16 = false, // Ignored (precision determined by TensorType)
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            // Cast interface float* to ElementType* for INPUT precision-specific processing
            const ElementType *Q_typed = reinterpret_cast<const ElementType *>(Q);
            const ElementType *K_typed = reinterpret_cast<const ElementType *>(K);
            const ElementType *V_typed = reinterpret_cast<const ElementType *>(V);

            // CRITICAL: Output stays as float* (GEMM kernels ALWAYS output FP32!)
            // Do NOT cast output to ElementType* - this breaks pointer arithmetic

            // Delegate to typed implementation
            return compute_typed(
                Q_typed, K_typed, V_typed, output, // output stays float*
                seq_len, n_heads, n_kv_heads, head_dim,
                causal, window_size,
                workspace_scores, workspace_buffer, workspace_context, workspace_mask,
                use_bf16, mpi_ctx, device_idx);
        }

    private:
        /**
         * @brief Type-specific attention implementation
         *
         * Internal method that works with ElementType* for INPUTS, float* for OUTPUT.
         * Called by public compute() method after casting input float* → ElementType*.
         *
         * CRITICAL: Output parameter is float* because:
         * 1. GEMM kernels ALWAYS output FP32 (not BF16/FP16/INT32)
         * 2. Workspace (scores, weights) is ALWAYS FP32Tensor
         * 3. Softmax operates on FP32
         * 4. Only inputs get type-specific processing via ActivationTraits
         */
        bool compute_typed(
            const ElementType *Q, const ElementType *K, const ElementType *V,
            float *output, // CRITICAL: float*, not ElementType*!
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal,
            int window_size,
            TensorBase *workspace_scores,
            TensorBase *workspace_buffer,
            TensorBase *workspace_context,
            TensorBase *workspace_mask,
            bool use_bf16,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            // Validate device
            if (device_idx != -1)
            {
                LOG_ERROR("[CPUAttentionT] compute: device_idx must be -1 (CPU), got " << device_idx);
                return false;
            }

            // Validate inputs
            if (!Q || !K || !V || !output)
            {
                LOG_ERROR("[CPUAttentionT] compute: null tensor data");
                return false;
            }

            if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || seq_len <= 0)
            {
                LOG_ERROR("[CPUAttentionT] compute: invalid dimensions");
                return false;
            }

            if (n_heads % n_kv_heads != 0)
            {
                LOG_ERROR("[CPUAttentionT] compute: n_heads (" << n_heads
                                                               << ") must be divisible by n_kv_heads (" << n_kv_heads << ")");
                return false;
            }

            // Allocate workspaces if not provided
            // CRITICAL: Workspaces are ALWAYS FP32, even for BF16/FP16 inputs!
            // Rationale: GEMM kernels produce FP32 output, softmax operates on FP32.
            // BF16/FP16 conversion happens inside GEMM kernels automatically.
            std::shared_ptr<TensorBase> owned_scores;
            std::shared_ptr<TensorBase> owned_buffer;
            std::shared_ptr<TensorBase> owned_context;

            if (!workspace_scores)
            {
                owned_scores = std::make_shared<FP32Tensor>(
                    std::vector<size_t>{static_cast<size_t>(n_heads * seq_len), static_cast<size_t>(seq_len)});
                workspace_scores = owned_scores.get();
            }

            // Get actual thread count for proper workspace sizing
            const int max_threads = omp_get_max_threads();
            const int actual_threads = (n_heads > 1) ? std::min(max_threads, n_heads) : 1;

            if (!workspace_buffer)
            {
                // Per-thread buffer: [actual_threads * seq_len * head_dim * 2] (Q_h + K_h/V_h)
                owned_buffer = std::make_shared<FP32Tensor>(
                    std::vector<size_t>{static_cast<size_t>(actual_threads * seq_len * head_dim * 2)});
                workspace_buffer = owned_buffer.get();
            }

            if (!workspace_context)
            {
                owned_context = std::make_shared<FP32Tensor>(
                    std::vector<size_t>{static_cast<size_t>(actual_threads * seq_len * head_dim)});
                workspace_context = owned_context.get();
            }

            // NOTE: Workspaces are ALWAYS FP32, even for BF16/FP16 inputs!
            // Rationale: GEMM kernels produce FP32 output, softmax operates on FP32.
            // Only final output gets converted to BF16/FP16 if needed.
            float *scores = workspace_scores->mutable_data();

            // 1. Convert Q, K, V to FP32 (required for multiply_activations_strided)
            // CRITICAL: ITensorGemm::multiply_activations_strided expects float* inputs!
            std::vector<float> Q_fp32(seq_len * n_heads * head_dim);
            std::vector<float> K_fp32(seq_len * n_kv_heads * head_dim);
            std::vector<float> V_fp32(seq_len * n_kv_heads * head_dim);

            // Convert ElementType → FP32 using type-specific conversion
            if constexpr (std::is_same_v<ElementType, float>)
            {
                // FP32: direct copy
                std::memcpy(Q_fp32.data(), Q, Q_fp32.size() * sizeof(float));
                std::memcpy(K_fp32.data(), K, K_fp32.size() * sizeof(float));
                std::memcpy(V_fp32.data(), V, V_fp32.size() * sizeof(float));
            }
            else if constexpr (std::is_same_v<TensorType, BF16Tensor>)
            {
                // BF16: convert using simd::bf16_to_fp32
                for (size_t i = 0; i < Q_fp32.size(); ++i)
                {
                    Q_fp32[i] = simd::bf16_to_fp32(Q[i]);
                }
                for (size_t i = 0; i < K_fp32.size(); ++i)
                {
                    K_fp32[i] = simd::bf16_to_fp32(K[i]);
                }
                for (size_t i = 0; i < V_fp32.size(); ++i)
                {
                    V_fp32[i] = simd::bf16_to_fp32(V[i]);
                }
            }
            else if constexpr (std::is_same_v<TensorType, FP16Tensor>)
            {
                // FP16: convert using simd::fp16_to_fp32
                for (size_t i = 0; i < Q_fp32.size(); ++i)
                {
                    Q_fp32[i] = simd::fp16_to_fp32(Q[i]);
                }
                for (size_t i = 0; i < K_fp32.size(); ++i)
                {
                    K_fp32[i] = simd::fp16_to_fp32(K[i]);
                }
                for (size_t i = 0; i < V_fp32.size(); ++i)
                {
                    V_fp32[i] = simd::fp16_to_fp32(V[i]);
                }
            }
            else if constexpr (std::is_same_v<TensorType, INT32Tensor>)
            {
                // INT32: convert using static_cast
                for (size_t i = 0; i < Q_fp32.size(); ++i)
                {
                    Q_fp32[i] = static_cast<float>(Q[i]);
                }
                for (size_t i = 0; i < K_fp32.size(); ++i)
                {
                    K_fp32[i] = static_cast<float>(K[i]);
                }
                for (size_t i = 0; i < V_fp32.size(); ++i)
                {
                    V_fp32[i] = static_cast<float>(V[i]);
                }
            }
            else if constexpr (std::is_same_v<TensorType, Q8_0Tensor>)
            {
                // Q8_0: dequantize blocks to FP32
                // Q8_0Tensor stores int8_t* but we need to cast through TensorBase to call to_fp32()
                // CRITICAL: Q is actually const int8_t*, but we need to treat the whole tensor
                // Create temporary Q8_0Tensor wrappers to use to_fp32() method

                // Calculate raw data size (34 bytes per block of 32 elements)
                size_t Q_n_elems = Q_fp32.size();
                size_t K_n_elems = K_fp32.size();
                size_t V_n_elems = V_fp32.size();

                size_t Q_n_blocks = (Q_n_elems + 31) / 32;
                size_t K_n_blocks = (K_n_elems + 31) / 32;
                size_t V_n_blocks = (V_n_elems + 31) / 32;

                size_t Q_n_bytes = Q_n_blocks * 34; // 34 bytes per Q8_0Block
                size_t K_n_bytes = K_n_blocks * 34;
                size_t V_n_bytes = V_n_blocks * 34;

                // Wrap raw Q8_0 data in Q8_0Tensor for dequantization
                std::vector<uint8_t> Q_raw(reinterpret_cast<const uint8_t *>(Q), reinterpret_cast<const uint8_t *>(Q) + Q_n_bytes);
                std::vector<uint8_t> K_raw(reinterpret_cast<const uint8_t *>(K), reinterpret_cast<const uint8_t *>(K) + K_n_bytes);
                std::vector<uint8_t> V_raw(reinterpret_cast<const uint8_t *>(V), reinterpret_cast<const uint8_t *>(V) + V_n_bytes);

                // CRITICAL: Q8_0Tensor requires 2D shape for to_fp32_via_blocks()
                // Shape: [seq_len * n_heads, head_dim] for Q, [seq_len * n_kv_heads, head_dim] for K/V
                Q8_0Tensor Q_tensor({static_cast<size_t>(seq_len * n_heads), static_cast<size_t>(head_dim)}, Q_raw);
                Q8_0Tensor K_tensor({static_cast<size_t>(seq_len * n_kv_heads), static_cast<size_t>(head_dim)}, K_raw);
                Q8_0Tensor V_tensor({static_cast<size_t>(seq_len * n_kv_heads), static_cast<size_t>(head_dim)}, V_raw);

                Q_tensor.to_fp32(Q_fp32.data());
                K_tensor.to_fp32(K_fp32.data());
                V_tensor.to_fp32(V_fp32.data());
            }
            else
            {
                // Unknown type - should never happen with proper ActivationTraits
                throw std::runtime_error("Unsupported tensor type for CPUAttentionT");
            }

            // 2. Broadcast K/V heads if needed (GQA/MQA)
            std::vector<float> K_broadcast, V_broadcast;
            const float *K_expanded = K_fp32.data();
            const float *V_expanded = V_fp32.data();

            if (n_heads != n_kv_heads)
            {
                K_broadcast.resize(seq_len * n_heads * head_dim);
                V_broadcast.resize(seq_len * n_heads * head_dim);

                // broadcast_kv_heads: FP32 → FP32 broadcast (no conversion)
                attention_utils::broadcast_kv_heads(
                    K_fp32.data(), K_broadcast.data(),
                    seq_len, n_heads, n_kv_heads, head_dim);
                attention_utils::broadcast_kv_heads(
                    V_fp32.data(), V_broadcast.data(),
                    seq_len, n_heads, n_kv_heads, head_dim);

                K_expanded = K_broadcast.data();
                V_expanded = V_broadcast.data();
            }

            // 3. Compute attention scores per head: Q @ K^T with fused scaling
            // Optimization 1: Use GEMM alpha parameter for attention scaling (1/sqrt(d_k))
            // Optimization 2: Use strided GEMM to eliminate Q/K head extraction (zero-copy)
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

            // Create GEMM kernel once (reused across heads) using ActivationTraits!
            auto gemm = Traits::create_activation_gemm();

#pragma omp parallel for if (n_heads > 1)
            for (int h = 0; h < n_heads; ++h)
            {
                float *scores_h = scores + h * seq_len * seq_len; // FP32 workspace

                // Strided Q@K^T with fused scaling
                // Q layout: [seq_len, n_heads, head_dim] (head dimension interleaved)
                // K layout: [seq_len, n_heads, head_dim] (head dimension interleaved)
                const float *Q_h = Q_fp32.data() + h * head_dim; // Q_fp32: already FP32
                const float *K_h = K_expanded + h * head_dim;    // K_expanded: already FP32

                const int lda = n_heads * head_dim; // Q: stride between rows (skip other heads)
                const int ldb = n_heads * head_dim; // K: stride between rows (skip other heads)
                const int ldc = seq_len;            // scores: contiguous [seq_len, seq_len]

                // Strided GEMM: scores = (1/sqrt(d_k)) * Q @ K^T
                // Note: Q_h and K_h are already FP32, no type conversion needed
                gemm->multiply_activations_strided(
                    Q_h,                        // FP32 Q head
                    K_h,                        // FP32 K head (after broadcast if GQA)
                    scores_h,                   // FP32 scores output
                    seq_len, seq_len, head_dim, // m=seq_len, n=seq_len, k=head_dim
                    lda, ldb, ldc,
                    true,    // transpose_B=true (K^T)
                    scale,   // alpha=1/sqrt(d_k) - FUSED SCALING!
                    0.0f,    // beta
                    nullptr, // mpi_ctx
                    -1);     // device_idx (CPU)
            }

            // 3. Apply softmax with optional causal masking
            // Softmax operates on FP32 workspaces (even for BF16/FP16 inputs)

#pragma omp parallel for if (n_heads > 1)
            for (int h = 0; h < n_heads; ++h)
            {
                float *scores_h = scores + h * seq_len * seq_len; // FP32 workspace

                // Apply FP32 softmax directly (workspaces are always FP32)
                // NO scaling (already fused in GEMM alpha parameter)
                primitives::softmax_row_major_fp32(scores_h, seq_len, seq_len, causal, 1.0f, true);
            }

            // 4. Compute context: weights @ V using strided GEMM (zero-copy multi-head)
            // Zero output once before parallel region
            // CRITICAL: Output is float* (4 bytes/element), not ElementType*!
            std::memset(output, 0, seq_len * n_heads * head_dim * sizeof(float));

            // Reuse GEMM kernel from Q@K^T computation

#pragma omp parallel for if (n_heads > 1)
            for (int h = 0; h < n_heads; ++h)
            {
                // NOTE: weights_h points to FP32 workspace (scores)
                const float *weights_h = scores + h * seq_len * seq_len;
                const float *V_h = V_expanded + h * head_dim; // V_expanded is float* after broadcast

                // CRITICAL: output is float* (not ElementType*), pointer arithmetic in floats
                float *output_h = output + h * head_dim; // First element of head h

                // Strided GEMM: context = weights @ V
                // V layout: [seq_len, n_heads, head_dim] (head dimension interleaved)
                // Output layout: [seq_len, n_heads, head_dim] (head dimension interleaved)
                const int lda = seq_len;            // weights: contiguous [seq_len, seq_len]
                const int ldb = n_heads * head_dim; // V: stride between rows (skip other heads)
                const int ldc = n_heads * head_dim; // output: stride between rows

                // Strided GEMM: context = weights @ V
                // Note: ITensorGemm interface uses float*, output_h is already float*
                gemm->multiply_activations_strided(
                    weights_h,                  // Already float*, no cast needed
                    V_h,                        // Already float*, no cast needed
                    output_h,                   // Already float*, no cast needed
                    seq_len, head_dim, seq_len, // m=seq_len, n=head_dim, k=seq_len
                    lda, ldb, ldc,
                    false,   // transpose_B=false (weights @ V, not V^T)
                    1.0f,    // alpha
                    0.0f,    // beta (output already zeroed)
                    nullptr, // mpi_ctx
                    -1);     // device_idx (CPU)
            }

            return true;
        }

        /**
         * @brief Batch attention computation (not yet implemented)
         */
        bool compute_batch(
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
            int device_idx = -1) override
        {
            LOG_ERROR("[CPUAttentionT] compute_batch: not yet implemented");
            return false;
        }

        /**
         * @brief Broadcast K/V from n_kv_heads to n_heads (for GQA/MQA)
         *
         * For GQA: Each KV head is replicated to multiple query heads
         * For MQA: Single KV head is replicated to all query heads
         * For MHA: No-op (n_heads == n_kv_heads)
         *
         * @param input K or V tensor [seq_len, n_kv_heads, head_dim]
         * @param output Broadcasted tensor [seq_len, n_heads, head_dim]
         * @param seq_len Sequence length
         * @param n_heads Number of query heads (target)
         * @param n_kv_heads Number of key/value heads (source)
         * @param head_dim Dimension per head
         */
        void broadcast_kv(
            const ElementType *input, ElementType *output,
            int seq_len, int n_heads, int n_kv_heads, int head_dim) const
        {
            // Use AttentionUtils to broadcast KV heads from n_kv_heads to n_heads
            // Note: AttentionUtils currently expects float*, so we cast
            // TODO: Make AttentionUtils template-based for better type safety
            attention_utils::broadcast_kv_heads(
                reinterpret_cast<const float *>(input),
                reinterpret_cast<float *>(output),
                seq_len, n_heads, n_kv_heads, head_dim);
        }
    };

    // Explicit instantiations (zero code duplication!)
    // These are defined in CPUAttentionT.cpp to avoid template bloat
    extern template class CPUAttentionT<FP32Tensor>;
    extern template class CPUAttentionT<BF16Tensor>;
    extern template class CPUAttentionT<FP16Tensor>;
    extern template class CPUAttentionT<INT32Tensor>;

} // namespace llaminar2
