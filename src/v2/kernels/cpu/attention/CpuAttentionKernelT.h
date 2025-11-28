/**
 * @file CpuAttentionKernelT.h
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

#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/SIMDHelpers.h"
#include "../CPUKernelBase.h"
#include "../primitives/ActivationTraits.h"
#include "../primitives/SoftmaxPrimitivesImpl.h"
#include "../../../pipelines/AttentionUtils.h"
#include "../../../utils/Logger.h"
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
     * @tparam TensorT Tensor type (FP32Tensor, BF16Tensor, FP16Tensor, INT32Tensor)
     *
     * Uses ActivationTraits<TensorT> for precision-specific operations:
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
    template <typename TensorT>
    class CpuAttentionKernelT : public ITensorAttention, public CPUKernelBase
    {
    public:
        // Infer ElementType from ActivationTraits (which knows the mapping)
        using ElementType = typename primitives::ActivationTraits<TensorT>::ElementType;
        using Traits = primitives::ActivationTraits<TensorT>;

        CpuAttentionKernelT() = default;
        ~CpuAttentionKernelT() override = default;

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
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            // Cast inputs to ElementType
            const ElementType *Q_typed = reinterpret_cast<const ElementType *>(Q);
            const ElementType *K_typed = reinterpret_cast<const ElementType *>(K);
            const ElementType *V_typed = reinterpret_cast<const ElementType *>(V);

            // Allocate workspaces if not provided
            std::shared_ptr<TensorBase> scores_alloc, buffer_alloc, context_alloc;

            float *scores_ptr = nullptr;
            float *buffer_ptr = nullptr;
            float *context_ptr = nullptr;
            const float *mask_ptr = nullptr;

            if (!workspace_scores)
            {
                scores_alloc = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(n_heads * seq_len * seq_len)});
                scores_ptr = scores_alloc->mutable_data();
            }
            else
            {
                scores_ptr = workspace_scores->mutable_data();
            }

            // buffer_ptr is no longer used (Virtual GQA)

            if (!workspace_context)
            {
                // Not strictly needed if writing directly to output, but kept for API compatibility
                // compute_typed writes directly to output
            }
            else
            {
                context_ptr = workspace_context->mutable_data();
            }

            if (workspace_mask)
            {
                mask_ptr = workspace_mask->data();
            }

            return compute_typed(
                Q_typed, K_typed, V_typed, output,
                seq_len, n_heads, n_kv_heads, head_dim,
                causal, window_size,
                scores_ptr, context_ptr,
                mask_ptr,
                device_idx,
                seq_len); // kv_len = seq_len for standard compute
        }

        /**
         * @brief Compute attention with separate sequence and KV lengths (Decoding)
         */
        bool compute_decode(
            const float *Q, const float *K, const float *V,
            float *output,
            int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal = false,
            int window_size = -1,
            TensorBase *workspace_scores = nullptr,
            TensorBase *workspace_buffer = nullptr,
            TensorBase *workspace_context = nullptr,
            TensorBase *workspace_mask = nullptr,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            const ElementType *Q_typed = reinterpret_cast<const ElementType *>(Q);
            const ElementType *K_typed = reinterpret_cast<const ElementType *>(K);
            const ElementType *V_typed = reinterpret_cast<const ElementType *>(V);

            // Allocate workspaces if not provided
            std::shared_ptr<TensorBase> scores_alloc;
            float *scores_ptr = nullptr;
            float *context_ptr = nullptr;
            const float *mask_ptr = nullptr;

            if (!workspace_scores)
            {
                scores_alloc = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(n_heads * seq_len * kv_len)});
                scores_ptr = scores_alloc->mutable_data();
            }
            else
            {
                scores_ptr = workspace_scores->mutable_data();
            }

            if (workspace_context)
            {
                context_ptr = workspace_context->mutable_data();
            }

            if (workspace_mask)
            {
                mask_ptr = workspace_mask->data();
            }

            return compute_typed(
                Q_typed, K_typed, V_typed, output,
                seq_len, n_heads, n_kv_heads, head_dim,
                causal, window_size,
                scores_ptr, context_ptr,
                mask_ptr,
                device_idx,
                kv_len);
        }

    private:
        /**
         * @brief Typed implementation of attention
         *
         * @param Q Input Q [seq_len, n_heads, head_dim]
         * @param K Input K [kv_len, n_kv_heads, head_dim]
         * @param V Input V [kv_len, n_kv_heads, head_dim]
         * @param output Output [seq_len, n_heads, head_dim] (always float*)
         * @param scores Workspace for attention scores [n_heads, seq_len, kv_len] (FP32)
         * @param buffer Workspace for KV broadcast [2, seq_len, n_heads, head_dim] (FP32/BF16/etc)
         * @param context Workspace for context (unused, writes to output)
         */
        bool compute_typed(
            const ElementType *Q, const ElementType *K, const ElementType *V,
            float *output,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal,
            int window_size,
            float *scores,
            float *context,
            const float *mask,
            int device_idx,
            int kv_len) const
        {
            // Validate device
            if (device_idx != -1)
            {
                LOG_ERROR("[CpuAttentionKernelT] compute: device_idx must be -1 (CPU), got " << device_idx);
                return false;
            }

            // Validate inputs
            if (!Q || !K || !V || !output)
            {
                LOG_ERROR("[CpuAttentionKernelT] compute: null tensor data");
                return false;
            }

            if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || seq_len <= 0 || kv_len <= 0)
            {
                LOG_ERROR("[CpuAttentionKernelT] compute: invalid dimensions");
                return false;
            }

            if (n_heads % n_kv_heads != 0)
            {
                LOG_ERROR("[CpuAttentionKernelT] compute: n_heads (" << n_heads
                                                                     << ") must be divisible by n_kv_heads (" << n_kv_heads << ")");
                return false;
            }

            // Create GEMM kernel once (reused across heads) using ActivationTraits!
            auto gemm = Traits::create_activation_gemm();

            // Optimized path for FP32, BF16, FP16 (Native & Mixed Precision)
            if constexpr (std::is_same_v<TensorT, FP32Tensor> ||
                          std::is_same_v<TensorT, BF16Tensor> ||
                          std::is_same_v<TensorT, FP16Tensor>)
            {
                ActivationFormat format = ActivationFormat::FP32;
                if constexpr (std::is_same_v<TensorT, BF16Tensor>)
                    format = ActivationFormat::BF16;
                else if constexpr (std::is_same_v<TensorT, FP16Tensor>)
                    format = ActivationFormat::FP16;

                // 1. Compute Q @ K^T -> Scores (FP32) + Mask + Softmax
                // No GQA broadcast needed! We handle strides virtually.
                const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
                const int heads_per_kv = n_heads / n_kv_heads;

                // Parallelism strategy:
                // 1. Decoding (seq_len < 128): Always parallelize over heads to hide latency/overhead.
                //    BLAS threading is inefficient for small matrices (M=1).
                // 2. Prefill (seq_len >= 128):
                //    - If we have enough heads to saturate ~50% of cores, parallelize over heads.
                //      (Single-threaded GEMM is efficient, avoids sync overhead).
                //    - If few heads (e.g. 1-4) on many cores, run sequentially to let BLAS use all cores.
                const int max_threads = omp_get_max_threads();
                bool parallelize_heads = true;
                if (seq_len >= 128 && n_heads * 2 < max_threads)
                {
                    parallelize_heads = false;
                }

                // 2. Scores @ V -> Output (FP32)
                std::memset(output, 0, seq_len * n_heads * head_dim * sizeof(float));

#pragma omp parallel for if (parallelize_heads)
                for (int h = 0; h < n_heads; ++h)
                {
                    // --- Step 1: Q @ K^T -> Scores ---
                    float *scores_h = scores + h * seq_len * kv_len;
                    const ElementType *Q_h = Q + h * head_dim;

                    // Virtual GQA: Map head h to kv_head
                    int kv_h = h / heads_per_kv;
                    const ElementType *K_h = K + kv_h * head_dim;

                    const int lda_k = n_heads * head_dim;
                    const int ldb_k = n_kv_heads * head_dim; // Stride of K is based on n_kv_heads
                    const int ldc_k = kv_len;

                    // Optimized path for decoding (seq_len == 1) on FP32
                    if (seq_len == 1 && std::is_same_v<TensorT, FP32Tensor>)
                    {
                        if constexpr (std::is_same_v<TensorT, FP32Tensor>)
                        {
                            // 1. Q @ K^T
                            for (int t = 0; t < kv_len; ++t)
                            {
                                const float *k_ptr = K_h + t * ldb_k;
                                float dot = 0.0f;
#pragma omp simd reduction(+ : dot)
                                for (int d = 0; d < head_dim; ++d)
                                {
                                    dot += Q_h[d] * k_ptr[d];
                                }
                                scores_h[t] = dot * scale;
                                if (mask)
                                    scores_h[t] += mask[t];
                            }

                            // 2. Softmax (Vectorized)
                            // Note: scale is already applied in step 1, so we pass 1.0f here.
                            // Causal masking is also handled in step 1 via the mask add.
                            llaminar2::primitives::softmax_row_fp32(
                                scores_h,
                                kv_len,
                                false, // causal
                                1.0f,  // scale
                                -1     // row_idx
                            );

                            // 3. Scores @ V
                            const float *weights_h = scores_h;
                            const ElementType *V_h = V + kv_h * head_dim;
                            float *output_h = output + h * head_dim;

                            const int ldb_v = n_kv_heads * head_dim;

                            // Initialize output to 0
                            std::memset(output_h, 0, head_dim * sizeof(float));

                            for (int t = 0; t < kv_len; ++t)
                            {
                                float s = weights_h[t];
                                const float *v_ptr = V_h + t * ldb_v;
#pragma omp simd
                                for (int d = 0; d < head_dim; ++d)
                                {
                                    output_h[d] += s * v_ptr[d];
                                }
                            }
                            continue; // Skip generic path
                        }
                    }

                    gemm->template multiply_with_softmax_strided_typed<ElementType, ElementType>(
                        Q_h, K_h, scores_h,
                        seq_len, kv_len, head_dim,
                        lda_k, ldb_k, ldc_k,
                        scale,
                        true, // transpose_B
                        1,    // softmax_axis
                        mask,
                        causal,
                        nullptr, -1, format);

                    // --- Step 2: Scores @ V -> Output ---
                    const float *weights_h = scores_h; // Reuse scores_h
                    const ElementType *V_h = V + kv_h * head_dim;
                    float *output_h = output + h * head_dim;

                    const int lda_v = kv_len;
                    const int ldb_v = n_kv_heads * head_dim; // Stride of V is based on n_kv_heads
                    const int ldc_v = n_heads * head_dim;

                    gemm->template multiply_activations_strided_typed<float, ElementType>(
                        weights_h, V_h, output_h,
                        seq_len, head_dim, kv_len,
                        lda_v, ldb_v, ldc_v,
                        false, 1.0f, 0.0f, nullptr, -1, format);
                }

                return true;
            }

            // 1. Convert Q, K, V to FP32 (required for multiply_activations_strided)
            // CRITICAL: ITensorGemm::multiply_activations_strided expects float* inputs!
            std::vector<float> Q_fp32(seq_len * n_heads * head_dim);
            std::vector<float> K_fp32(kv_len * n_kv_heads * head_dim);
            std::vector<float> V_fp32(kv_len * n_kv_heads * head_dim);

            // Convert ElementType -> FP32 using type-specific conversion
            if constexpr (std::is_same_v<ElementType, float>)
            {
                // FP32: direct copy
                std::memcpy(Q_fp32.data(), Q, Q_fp32.size() * sizeof(float));
                std::memcpy(K_fp32.data(), K, K_fp32.size() * sizeof(float));
                std::memcpy(V_fp32.data(), V, V_fp32.size() * sizeof(float));
            }
            else if constexpr (std::is_same_v<TensorT, BF16Tensor>)
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
            else if constexpr (std::is_same_v<TensorT, FP16Tensor>)
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
            else if constexpr (std::is_same_v<TensorT, Q8_1Tensor>)
            {
                // Q8_1: decode blocks
                // Q is Q8_1Block*
                // Q_fp32 is float*
                // Each block decodes to 32 floats
                size_t n_blocks = Q_fp32.size() / 32;
                for (size_t i = 0; i < n_blocks; ++i)
                {
                    Q8_1Tensor::decodeBlock(Q[i], Q_fp32.data() + i * 32);
                }

                n_blocks = K_fp32.size() / 32;
                for (size_t i = 0; i < n_blocks; ++i)
                {
                    Q8_1Tensor::decodeBlock(K[i], K_fp32.data() + i * 32);
                }

                n_blocks = V_fp32.size() / 32;
                for (size_t i = 0; i < n_blocks; ++i)
                {
                    Q8_1Tensor::decodeBlock(V[i], V_fp32.data() + i * 32);
                }
            }
            else if constexpr (std::is_same_v<TensorT, INT32Tensor>)
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
            else if constexpr (std::is_same_v<TensorT, Q8_0Tensor>)
            {
                // Q8_0: dequantize blocks to FP32
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
                throw std::runtime_error("Unsupported tensor type for CpuAttentionKernelT");
            }

            // 2. Compute attention scores per head: Q @ K^T with fused scaling
            // No GQA broadcast needed! We handle strides virtually.
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            const int heads_per_kv = n_heads / n_kv_heads;

            // GEMM kernel already created above

            // Parallelism strategy:
            // 1. Decoding (seq_len < 128): Always parallelize over heads to hide latency/overhead.
            //    BLAS threading is inefficient for small matrices (M=1).
            // 2. Prefill (seq_len >= 128):
            //    - If we have enough heads to saturate ~50% of cores, parallelize over heads.
            //      (Single-threaded GEMM is efficient, avoids sync overhead).
            //    - If few heads (e.g. 1-4) on many cores, run sequentially to let BLAS use all cores.
            const int max_threads = omp_get_max_threads();
            bool parallelize_heads = true;
            if (seq_len >= 128 && n_heads * 2 < max_threads)
            {
                parallelize_heads = false;
            }

#pragma omp parallel for if (parallelize_heads)
            for (int h = 0; h < n_heads; ++h)
            {
                float *scores_h = scores + h * seq_len * seq_len; // FP32 workspace

                // Strided Q@K^T with fused scaling
                const float *Q_h = Q_fp32.data() + h * head_dim; // Q_fp32: already FP32

                // Virtual GQA: Map head h to kv_head
                int kv_h = h / heads_per_kv;
                const float *K_h = K_fp32.data() + kv_h * head_dim; // K_fp32: already FP32

                const int lda = n_heads * head_dim;    // Q: stride between rows (skip other heads)
                const int ldb = n_kv_heads * head_dim; // K: stride between rows (skip other heads)
                const int ldc = seq_len;               // scores: contiguous [seq_len, seq_len]

                // Strided GEMM: scores = (1/sqrt(d_k)) * Q @ K^T
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

            if (mask)
            {
#pragma omp parallel for if (n_heads > 1)
                for (int h = 0; h < n_heads; ++h)
                {
                    float *scores_h = scores + h * seq_len * seq_len;
                    attention_utils::apply_attention_mask(scores_h, mask, seq_len, seq_len);
                }
            }

            // 3. Apply softmax with optional causal masking
            // Softmax operates on FP32 workspaces (even for BF16/FP16 inputs)
            // Softmax is lightweight, parallelize even with few heads

#pragma omp parallel for if (n_heads > 1)
            for (int h = 0; h < n_heads; ++h)
            {
                float *scores_h = scores + h * seq_len * seq_len; // FP32 workspace

                // Apply FP32 softmax directly (workspaces are always FP32)
                primitives::softmax_row_major_fp32(scores_h, seq_len, seq_len, causal, 1.0f, true);
            }

            // 4. Compute context: weights @ V using strided GEMM (zero-copy multi-head)
            // Zero output once before parallel region
            std::memset(output, 0, seq_len * n_heads * head_dim * sizeof(float));

            // Reuse GEMM kernel from Q@K^T computation
            // Reuse parallelism strategy

#pragma omp parallel for if (parallelize_heads)
            for (int h = 0; h < n_heads; ++h)
            {
                // NOTE: weights_h points to FP32 workspace (scores)
                const float *weights_h = scores + h * seq_len * seq_len;

                // Virtual GQA: Map head h to kv_head
                int kv_h = h / heads_per_kv;
                const float *V_h = V_fp32.data() + kv_h * head_dim; // V_fp32 is float*

                // CRITICAL: output is float* (not ElementType*), pointer arithmetic in floats
                float *output_h = output + h * head_dim; // First element of head h

                // Strided GEMM: context = weights @ V
                const int lda = seq_len;               // weights: contiguous [seq_len, seq_len]
                const int ldb = n_kv_heads * head_dim; // V: stride between rows (skip other heads)
                const int ldc = n_heads * head_dim;    // output: stride between rows

                // Strided GEMM: context = weights @ V
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
         * @brief Batch attention computation
         */
    public:
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
            // Allocate workspaces if not provided
            std::shared_ptr<TensorBase> scores_alloc, buffer_alloc, context_alloc;
            float *scores_ptr = nullptr;
            float *buffer_ptr = nullptr;
            float *context_ptr = nullptr;

            // Per-sequence scores: [batch_size, n_heads, seq_len, seq_len]
            // Flattened: batch_size * n_heads * seq_len * seq_len
            if (!workspace_scores)
            {
                scores_alloc = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(batch_size * n_heads * seq_len * seq_len)});
                scores_ptr = scores_alloc->mutable_data();
            }
            else
            {
                scores_ptr = workspace_scores->mutable_data();
            }

            // buffer_ptr is no longer used (Virtual GQA)

            const float *mask_ptr = nullptr;
            const int total_tokens = batch_size * seq_len;
            if (workspace_mask)
            {
                mask_ptr = workspace_mask->data();
            }

            // Allocate mask tile buffer once for reuse across all batch items
            // The combined mask is [total_tokens, total_tokens], but compute_typed expects [seq_len, seq_len]
            // We extract the relevant [seq_len, seq_len] block per batch item
            std::vector<float> mask_tile;
            if (mask_ptr)
            {
                mask_tile.resize(seq_len * seq_len);
            }

            // Cast inputs to ElementType FIRST (before any pointer arithmetic)
            // This is CRITICAL: Q/K/V are void* in disguise, pointing to native ElementType data
            // For FP32: ElementType=float, no-op cast
            // For BF16: ElementType=uint16_t, must cast before arithmetic to avoid wrong stride
            const ElementType *Q_typed = reinterpret_cast<const ElementType *>(Q);
            const ElementType *K_typed = reinterpret_cast<const ElementType *>(K);
            const ElementType *V_typed = reinterpret_cast<const ElementType *>(V);

            // Loop over batch items
            // Parallelize over batch if batch_size is large enough
            // But compute_typed already parallelizes over heads.
            // For small batch size (e.g. 1-8), sequential loop is fine.
            for (int b = 0; b < batch_size; ++b)
            {
                // Offsets in ELEMENT units (ElementType, not float!)
                // Q, K, V: [batch_size, seq_len, n_heads/n_kv_heads, head_dim] (flattened)
                // Stride per batch: seq_len * n_heads * head_dim
                size_t q_offset = b * seq_len * n_heads * head_dim;
                size_t k_offset = b * seq_len * n_kv_heads * head_dim;
                size_t v_offset = b * seq_len * n_kv_heads * head_dim;
                size_t out_offset = b * seq_len * n_heads * head_dim;

                // Scores: [batch_size, n_heads, seq_len, seq_len] (always float)
                size_t scores_offset = b * n_heads * seq_len * seq_len;

                // Extract per-sequence mask tile from combined batch mask
                // Combined mask: [total_tokens, total_tokens] where total_tokens = batch_size * seq_len
                // For batch item b, tokens span [b*seq_len : (b+1)*seq_len]
                // Extract the [seq_len, seq_len] block at rows [b*seq_len : (b+1)*seq_len], cols [b*seq_len : (b+1)*seq_len]
                const float *mask_slice = nullptr;
                if (mask_ptr)
                {
                    const int row_offset = b * seq_len;
                    const int col_offset = b * seq_len;

                    // Copy the relevant block from combined mask to tile buffer
                    for (int i = 0; i < seq_len; ++i)
                    {
                        for (int j = 0; j < seq_len; ++j)
                        {
                            const int global_row = row_offset + i;
                            const int global_col = col_offset + j;
                            mask_tile[i * seq_len + j] = mask_ptr[global_row * total_tokens + global_col];
                        }
                    }

                    mask_slice = mask_tile.data();
                }

                // Pointer arithmetic in ElementType units (already cast above)
                const ElementType *Q_slice = Q_typed + q_offset;
                const ElementType *K_slice = K_typed + k_offset;
                const ElementType *V_slice = V_typed + v_offset;

                bool success = compute_typed(
                    Q_slice,
                    K_slice,
                    V_slice,
                    output + out_offset,
                    seq_len, n_heads, n_kv_heads, head_dim,
                    causal, window_size,
                    scores_ptr + scores_offset,
                    nullptr, // context not used
                    mask_slice,
                    device_idx,
                    seq_len); // kv_len = seq_len for batch compute

                if (!success)
                    return false;
            }

            return true;
        }
    };

    // Explicit instantiations (zero code duplication!)
    // These are defined in CpuAttentionKernelT.cpp to avoid template bloat
    extern template class CpuAttentionKernelT<FP32Tensor>;
    extern template class CpuAttentionKernelT<BF16Tensor>;
    extern template class CpuAttentionKernelT<FP16Tensor>;
} // namespace llaminar2
