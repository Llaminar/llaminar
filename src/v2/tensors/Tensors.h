/**
 * @file Tensors.h
 * @brief Minimal tensor interface with device affinity
 *
 * ============================================================================
 * INTERFACE HIERARCHY AND USAGE
 * ============================================================================
 *
 * This file defines several key interfaces that tensor classes implement:
 *
 * 1. **IActivationTensor** - For mutable activation tensors
 *    - Implemented by: FP32Tensor, FP16Tensor, BF16Tensor, INT32Tensor, Q8_1Tensor
 *    - NOT implemented by: Quantized weight tensors (IQ4_NL, Q6_K, Q4_K, etc.)
 *    - Usage: Pipeline casts to IActivationTensor* for RMSNorm, RoPE, SwiGLU ops
 *    - Key methods: createRoPE(), createSwiGLU(), applyRMSNorm(), to_int8_activation_pack()
 *
 * 2. **ITensorGemmTileDataProvider** - Block-wise FP32 decode interface
 *    - Implemented by: ALL tensor types (quantized weights + activation tensors)
 *    - Usage: Generic QuantizedGemmKernel, auto-tuner, tile-based GEMM
 *    - Key methods: decode_block_at(), block_size(), decoder_rows()/decoder_cols()
 *    - Enables uniform GEMM kernel that works across all quantization formats
 *
 * 3. **IQ8_0Decodable** - On-the-fly conversion to Q8_0 symmetric quantization
 *    - Implemented by: Quantized weight tensors + activation tensors (for testing)
 *    - Usage: Weight caching, INT8×INT8 GEMM intermediate format
 *    - Key method: decode_to_q8_0() - converts arbitrary quantized block → Q8_0
 *
 * 4. **IQ8_1Decodable** - On-the-fly conversion to Q8_1 asymmetric quantization
 *    - Implemented by: Activation tensors only (FP32, FP16, BF16, INT32, Q8_1)
 *    - Usage: INT8 GEMM activations (AVX512-VNNI, CUDA Tensor Cores)
 *    - Key method: decode_to_q8_1() - quantizes activation row to Q8_1 with pre-computed sum
 *    - Optimization: "Quantize once, use many times" for multi-head attention
 *
 * ============================================================================
 * TENSOR CLASS SUMMARY
 * ============================================================================
 *
 * **Activation Tensors** (implements IActivationTensor):
 * - FP32Tensor:  32-bit float (baseline, full precision)
 * - FP16Tensor:  IEEE 754 half-precision (2× compression, GPU-optimized)
 * - BF16Tensor:  Brain Float 16 (2× compression, FP32 range, CPU/GPU hardware support)
 * - INT32Tensor: 32-bit integer accumulator (INT8 GEMM pipeline intermediate)
 * - Q8_1Tensor:  8-bit asymmetric quantization with sum (activation format)
 *
 * **Quantized Weight Tensors** (read-only, implements ITensorGemmTileDataProvider + IQ8_0Decodable):
 * - IQ4_NLTensor:   4.5 bpw (7.1× compression, best 4-bit quality)
 * - IQ4_XSTensor:   4.5 bpw (alternative 4-bit scheme)
 * - Q8_0Tensor:     8.5 bpw (symmetric quantization)
 * - Q4_0Tensor:     4.5 bpw (symmetric 4-bit)
 * - Q4_1Tensor:     5.0 bpw (asymmetric 4-bit)
 * - Q5_0Tensor:     5.5 bpw (symmetric 5-bit)
 * - Q5_1Tensor:     6.0 bpw (asymmetric 5-bit)
 * - Q6_KTensor:     6.6 bpw (6-bit k-quant)
 * - Q2_KTensor:     2.6 bpw (2-bit k-quant)
 * - Q3_KTensor:     3.4 bpw (3-bit k-quant)
 * - Q4_KTensor:     4.5 bpw (4-bit k-quant)
 * - Q5_KTensor:     5.5 bpw (5-bit k-quant)
 * - Q8_KTensor:     8.5 bpw (8-bit k-quant)
 * - IQ2_XXSTensor:  2.1 bpw (ultra-low bit 2-bit)
 * - IQ2_XSTensor:   2.3 bpw (improved 2-bit)
 * - IQ3_XXSTensor:  3.1 bpw (ultra-low bit 3-bit)
 * - IQ2_STensor:    2.5 bpw (signed 2-bit)
 * - IQ3_STensor:    3.4 bpw (signed 3-bit)
 * - IQ1_STensor:    1.6 bpw (ultra-compressed 1-bit)
 * - IQ1_MTensor:    1.9 bpw (modified 1-bit)
 *
 * **Special Tensor** (hybrid):
 * - INT8Tensor: Dequantized 8-bit weights (4× compression vs FP32, AVX512-VNNI/CUDA INT8)
 *               Implements ITensorGemmTileDataProvider but NOT IActivationTensor
 *
 * Key design principles:
 * - Per-tensor device placement (not per-rank)
 * - Lazy host↔device synchronization
 * - Direct kernel creation (no operator layer)
 *
 * @author David Sanftenberg
 */

#pragma once

#include "TensorKernels.h"
#include "FP16Utils.h"
#include "BlockStructures.h" // Must be included BEFORE SIMDHelpers.h
#include "SIMDHelpers.h"
#include "AlignedVector.h"
#include <vector>
#include <memory>
#include <cstddef>
#include <cstdint>

namespace llaminar2
{
    // All block structures are now defined in BlockStructures.h
    // This eliminates circular dependencies between Tensors.h and SIMDHelpers.h

    /**
     * @brief Tensor data type
     */
    enum class TensorType
    {
        FP32,    // 32-bit float
        BF16,    // 16-bit bfloat
        FP16,    // 16-bit float
        INT8,    // 8-bit integer (dequantized for AVX512-VNNI/CUDA INT8 GEMM)
        INT32,   // 32-bit integer accumulator (for INT8 GEMM results)
        IQ4_NL,  // 4-bit quantized (non-linear)
        IQ4_XS,  // 4-bit quantized (extra-small IQ)
        Q8_0,    // 8-bit quantized (weights)
        Q8_1,    // 8-bit quantized with pre-computed sum (intermediate activation format)
        Q4_0,    // 4-bit quantized
        Q4_1,    // 4-bit quantized with min
        Q5_0,    // 5-bit quantized
        Q5_1,    // 5-bit quantized with min
        Q6_K,    // 6-bit K-quant
        Q2_K,    // 2-bit K-quant
        Q5_K,    // 5-bit K-quant
        Q3_K,    // 3-bit K-quant
        Q4_K,    // 4-bit K-quant
        Q8_K,    // 8-bit K-quant
        IQ2_XXS, // 2-bit extra-extra-small IQ
        IQ2_XS,  // 2-bit extra-small IQ
        IQ3_XXS, // 3-bit extra-extra-small IQ
        IQ2_S,   // 2-bit small IQ
        IQ3_S,   // 3-bit small IQ
        IQ1_S,   // 1-bit small IQ
        IQ1_M    // 1-bit medium IQ
    };

    struct ActivationPack
    {
        std::vector<int8_t> data;
        std::vector<float> row_scales;
        int rows = 0;
        int cols = 0;

        size_t element_count() const { return data.size(); }
    };

    /**
     * @brief Interface for activation tensors that support kernel creation and in-place operations
     *
     * **Purpose**: Marks tensor types that represent mutable activation data (not read-only weights).
     * Activation tensors can create computation kernels and support in-place transformations
     * like RMSNorm, RoPE, SwiGLU, and Softmax.
     *
     * **Implementors**: FP32Tensor, BF16Tensor, FP16Tensor, INT32Tensor, Q8_1Tensor
     * **Non-implementors**: Quantized weight tensors (IQ4_NL, Q6_K, Q4_K, etc.)
     *
     * **Rationale**:
     * - Kernels operate on ACTIVATION buffers (hidden states, Q/K/V, intermediate results)
     * - Kernel precision must match ACTIVATION precision, not weight precision
     * - Weights are read-only and should not create kernels (architectural invariant)
     *
     * **Usage in pipelines**:
     * - `Qwen2Pipeline::computeAttention()`: Casts current_hidden_ to IActivationTensor* for RMSNorm
     * - `Qwen2Pipeline::computeFFN()`: Uses IActivationTensor* for SwiGLU operations
     * - Allows pipeline to work with any activation precision (FP32/FP16/BF16/INT32)
     *
     * **Key methods**:
     * - `createRoPE()`, `createSwiGLU()`, `createSoftmax()`, `createRMSNorm()`, `createAttention()`
     * - `applyRMSNorm()`, `applyRoPE()`: In-place transformations
     * - `to_int8_activation_pack()`: Per-row quantization for INT8 GEMM activations
     * - `from_int32_with_scales()`: Dequantization from INT32 accumulator (INT8 pipeline)
     */
    class IActivationTensor
    {
    public:
        virtual ~IActivationTensor() = default;

        // Kernel creation (only for activation operations)
        virtual std::unique_ptr<ITensorRoPE> createRoPE() = 0;
        virtual std::unique_ptr<ITensorSwiGLU> createSwiGLU() = 0;
        virtual std::unique_ptr<ITensorSoftmax> createSoftmax() = 0;
        virtual std::unique_ptr<ITensorRMSNorm> createRMSNorm() = 0;
        virtual std::unique_ptr<ITensorAttention> createAttention() = 0;

        /**
         * @brief Quantize activations to INT8 with per-row scales
         * @param rows Number of rows (M dimension)
         * @param cols Number of columns (K dimension)
         * @return Packed activation data and per-row scales
         */
        virtual ActivationPack to_int8_activation_pack(int rows, int cols) const = 0;

        /**
         * @brief Apply RMSNorm in-place (native precision, no conversion overhead)
         *
         * Each tensor type implements this using its native precision method:
         * - FP32Tensor: apply() with FP32 buffers
         * - BF16Tensor: apply_bf16() with BF16 buffers
         * - FP16Tensor: apply_fp16() with FP16 buffers
         * - INT32Tensor: apply_int32_to_int8() with INT32→INT8 requantization
         *
         * @param gamma RMSNorm scale parameters [d_model] (always FP32)
         * @param seq_len Sequence length
         * @param d_model Model dimension
         * @param eps Epsilon for numerical stability
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index for kernel execution
         *
         * @return true on success, false on failure
         */
        virtual bool applyRMSNorm(
            const float *gamma,
            int seq_len,
            int d_model,
            float eps = 1e-6f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;

        /**
         * @brief Apply rotary position embeddings in-place (native precision)
         *
         * Each tensor type implements this using its native precision method:
         * - FP32Tensor: apply() with FP32 buffers
         * - BF16Tensor: apply_bf16() with BF16 buffers (faster, negligible loss)
         * - FP16Tensor: apply_fp16() with FP16 buffers
         * - INT32Tensor: Not supported (RoPE is activation operation, not quantized)
         *
         * @param Q Query tensor (this tensor, modified in-place)
         * @param K Key tensor (separate tensor, modified in-place)
         * @param position_ids Position indices [seq_len] (int32)
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads (GQA support)
         * @param head_dim Dimension per head
         * @param rope_theta RoPE frequency base (10000.0 for LLaMA, 1000000.0 for Qwen2.5)
         * @param use_bf16 Hint to use BF16 internally (tensor decides based on native type)
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index for kernel execution
         *
         * @return true on success, false on failure
         *
         * @note Q is this activation tensor, K is passed as parameter
         */
        virtual bool applyRoPE(
            float *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta = 10000.0f,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;

        /**
         * @brief Populate this activation tensor directly from INT32 accumulators
         *
         * @param accum Pointer to INT32 accumulator buffer in row-major order [rows, cols]
         * @param rows Number of rows (M dimension)
         * @param cols Number of columns (N dimension)
         * @param row_scales Per-row scales (length = rows). May be nullptr to treat as 1.0f
         * @param col_scales Per-column scales (length = cols). May be nullptr to treat as 1.0f
         * @param bias Optional bias vector (length = cols). May be nullptr if no bias
         * @return true on success, false if tensor cannot represent the requested shape/format
         */
        virtual bool from_int32_with_scales(
            const int32_t *accum,
            int rows,
            int cols,
            const float *row_scales,
            const float *col_scales,
            const float *bias = nullptr) = 0;
    };

    /**
     * @brief Interface for tensors that can be decoded to Q8_0 format (8-bit symmetric quantization)
     *
     * **Purpose**: Enables on-the-fly conversion of quantized weight tensors to Q8_0 intermediate format
     * for integer GEMM operations (INT8×INT8 → INT32). This is used for weight caching and
     * heterogeneous precision GEMM where weights and activations have different quantization formats.
     *
     * **Implementors**:
     * - Quantized weight tensors: IQ4_NL, Q6_K, Q4_K, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, etc.
     * - Activation tensors: FP32Tensor, FP16Tensor, BF16Tensor (for testing/compatibility)
     *
     * **Usage**:
     * - Weight caching systems that convert arbitrary quantized formats → Q8_0 for uniform GEMM
     * - INT8 GEMM kernels that want standardized 8-bit weight format
     * - `TensorBase::to_q8_0()` default implementation uses this interface
     *
     * **Q8_0 Format** (GGUF standard):
     * - 32 elements per block
     * - 1 FP16 scale per block
     * - Symmetric quantization: value = scale * int8_value
     * - No zero-point (always centered at 0)
     */
    class IQ8_0Decodable
    {
    public:
        virtual ~IQ8_0Decodable() = default;

        /**
         * @brief Decode a single Q8_0 block from this tensor
         * @param row_idx Row index in the tensor
         * @param k_block_offset Block offset along K dimension (32 elements per block)
         * @param output Pointer to output Q8_0Block (32 int8 values + 1 FP16 scale)
         */
        virtual void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const = 0;
    };

    /**
     * @brief Interface for tensors that can be decoded to Q8_1 format (8-bit asymmetric quantization with sum)
     *
     * **Purpose**: Enables on-the-fly quantization of activation tensors to Q8_1 intermediate format
     * for integer GEMM operations. Q8_1 pre-computes per-block sums during quantization to eliminate
     * expensive horizontal reductions in the GEMM inner loop.
     *
     * **Implementors**: FP32Tensor, FP16Tensor, BF16Tensor, INT32Tensor, Q8_1Tensor
     * **Non-implementors**: Quantized weight tensors (read-only, don't support activation quantization)
     *
     * **Q8_1 Format** (GGUF standard):
     * - 32 elements per block
     * - 1 FP16 scale (d) per block
     * - 1 FP16 sum (s = d × Σqs[i]) per block (KEY DIFFERENCE from Q8_0)
     * - Asymmetric quantization support via sum term
     *
     * **Optimization Strategy** ("quantize once, use many times"):
     * - Activation row quantized to Q8_1 once → cached in registers/L1
     * - GEMM K-loop uses pre-computed sum → no horizontal reduction overhead
     * - Matches CUDA llama.cpp pattern for INT8×INT8 GEMM
     *
     * **Usage**:
     * - INT8 GEMM kernels (AVX512-VNNI, CUDA INT8 Tensor Cores)
     * - OneDNN matmul with INT8 activations
     * - Cached activation quantization for multi-head attention
     */
    class IQ8_1Decodable
    {
    public:
        virtual ~IQ8_1Decodable() = default;

        /**
         * @brief Get a pointer to a Q8_1 block from this tensor
         *
         * For Q8_1Tensor: Returns direct pointer (zero-cost)
         * For FP32/FP16/BF16: Quantizes on-the-fly into thread-local buffer, returns pointer to it
         *
         * @param row_idx Row index in the tensor
         * @param k_block_offset Block offset along K dimension (32 elements per block)
         * @return const Q8_1Block* Pointer to the Q8_1 block (valid until next call on same thread)
         */
        virtual const Q8_1Block *decode_to_q8_1(size_t row_idx, size_t k_block_offset) const = 0;
    };

    /**
     * @brief Abstract tensor interface
     */
    class TensorBase : public std::enable_shared_from_this<TensorBase>
    {
    public:
        virtual ~TensorBase() = default;

        // Shape and type
        virtual const std::vector<size_t> &shape() const = 0;
        virtual TensorType native_type() const = 0;

        // Device affinity
        virtual int device_index() const = 0;        // -1 = host, ≥0 = device index from DeviceManager
        virtual bool set_device(int device_idx) = 0; // Upload to device
        virtual bool is_on_device(int device_idx) const = 0;

        // Data access (fallback for non-fused operations)
        // NOTE: Returns pointer in native type - use convertTo<T>() for type conversion
        virtual const float *data() const = 0; // Returns host pointer (syncs from device if needed)
        virtual float *mutable_data() = 0;     // Returns host pointer, marks dirty

        // Device transfers (Phase 4.2)
        virtual bool copyFrom(const TensorBase *src) = 0; // Copy data from another tensor (handles device transfers)

        // Kernel creation (only for weight matrices - GEMM)
        // NOTE: RoPE, SwiGLU, Softmax, RMSNorm, Attention moved to IActivationTensor
        virtual std::unique_ptr<ITensorGemm> createGemm() = 0;

        // ===== Generic Type Conversion API =====

        /**
         * @brief Convert tensor data to requested type
         * @tparam T Destination type: float (FP32), uint16_t (FP16/BF16 via format param), int8_t, int32_t
         * @param dst Destination buffer (must be pre-allocated)
         * @param format For uint16_t: TensorType::FP16 or TensorType::BF16 (default BF16)
         * @note Zero-copy when T matches native type, otherwise converts
         * @note Usage: tensor.to<float>(dst), tensor.to<int8_t>(dst), tensor.to<uint16_t>(dst, TensorType::FP16)
         */
        template <typename T>
        void to(T *dst, TensorType format = TensorType::BF16) const;

        // ===== Legacy Format Conversion Methods (implemented via convertTo<T>) =====

        /**
         * @brief Convert entire tensor to FP32 format
         * @param dst Destination buffer (must be pre-allocated with element_count() * sizeof(float))
         * @note For FP32 tensors, this is a memcpy. For quantized tensors, this dequantizes.
         */
        virtual void to_fp32(float *dst) const = 0;

        /**
         * @brief Convert entire tensor to BF16 format
         * @param dst Destination buffer (must be pre-allocated with element_count() * sizeof(uint16_t))
         * @note For BF16 tensors, this is a memcpy. For others, converts via FP32 intermediate.
         */
        virtual void to_bf16(uint16_t *dst) const = 0;

        /**
         * @brief Convert entire tensor to FP16 format
         * @param dst Destination buffer (must be pre-allocated with element_count() * sizeof(uint16_t))
         */
        virtual void to_fp16(uint16_t *dst) const = 0;

        /**
         * @brief Convert tensor to blocked int8 with per-block scale factors
         * @param dst_int8 Destination for int8 values (element_count() bytes)
         * @param dst_scales Destination for per-block scales (ceil(element_count()/block_size) floats)
         * @param block_size Elements per block (default 32, typical range 32-64)
         *
         * @note Enables AVX512-VNNI/DP4A int8 dot products with block-floating-point quantization
         * @note Scales stored as inverse (1.0/scale) for faster dequantization
         */
        virtual void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const = 0;

        /**
         * @brief Convert 2D tensor to INT8 with per-column and per-row scales
         * @param dst_int8 Destination for int8 values (rows * cols bytes)
         * @param dst_col_scales Destination for per-column scales (cols floats)
         * @param dst_row_scales Destination for per-row scales (rows floats, optional)
         * @return true if successful, false if tensor is not 2D or conversion not supported
         *
         * @note Uses ITensorGemmTileDataProvider interface to decode quantized formats directly to INT8
         * @note Avoids double-quantization error (GGUF → FP32 → INT8)
         * @note For weight matrices in INT8 GEMM operations
         */
        virtual bool to_int8_perchannel(int8_t *dst_int8,
                                        float *dst_col_scales,
                                        float *dst_row_scales = nullptr) const = 0;

        /**
         * @brief Convert a single row to FP32 (efficient for row-wise access patterns)
         * @param row_idx Row index to convert
         * @param buffer Destination buffer (must be pre-allocated with row_size * sizeof(float))
         */
        virtual void to_fp32_row(size_t row_idx, float *buffer) const = 0;

        /**
         * @brief Convert an arbitrary span to FP32 (efficient for slicing)
         * @param offset Starting element offset
         * @param count Number of elements to convert
         * @param buffer Destination buffer (must be pre-allocated with count * sizeof(float))
         */
        virtual void to_fp32_span(size_t offset, size_t count, float *buffer) const = 0;

        /**
         * @brief Convert tensor to Q8_0 format (blocked int8 with FP16 scales)
         * @param dst Destination buffer (must be pre-allocated with element_count()/32 * sizeof(Q8_0Block))
         * @note Implemented in TensorBase for all tensor types using ITensorGemmTileDataProvider or FP32 fallback
         * @note OpenMP parallelized for efficiency
         */
        void to_q8_0(Q8_0Block *dst) const;

    protected:
        ActivationPack pack_activation_rows_to_int8(int rows, int cols) const;

        /**
         * @brief Helper method for quantized tensors that implement ITensorGemmTileDataProvider
         * @param dst Destination FP32 buffer
         * @note This leverages the existing decode_block_at() method for block-quantized formats
         */
        void to_fp32_via_blocks(float *dst) const;

        /**
         * @brief Helper method for quantized tensors that implement ITensorGemmTileDataProvider
         *        Converts directly to INT8 with per-channel quantization
         * @param dst_int8 Destination INT8 buffer
         * @param dst_col_scales Destination for per-column scales
         * @param dst_row_scales Destination for per-row scales (optional)
         * @return true if successful, false if not 2D or not ITensorGemmTileDataProvider
         * @note Avoids double-quantization by decoding blocks directly to FP32 temporarily,
         *       then quantizing to INT8 with per-channel scales
         */
        bool to_int8_perchannel_via_blocks(int8_t *dst_int8,
                                           float *dst_col_scales,
                                           float *dst_row_scales = nullptr) const;

        /**
         * @brief Get total element count (product of all dimensions)
         */
        size_t element_count() const
        {
            size_t count = 1;
            for (size_t dim : shape())
            {
                count *= dim;
            }
            return count;
        }

    public:
        // ===== Tensor View Support =====

        /**
         * @brief Check if this tensor is a view of another tensor
         * @return true if this is a view, false if this owns its data
         */
        virtual bool is_view() const { return false; }

        /**
         * @brief Create a view into this tensor with a different shape
         *
         * Views share the underlying data buffer but present a different logical shape.
         * Useful for:
         * - Slicing tensors (e.g., first N tokens from pre-allocated buffer)
         * - Reshaping without copying
         * - Batch dimension manipulation
         *
         * @param new_shape The shape for the view
         * @param offset Offset in elements from the start of this tensor's data
         * @return Shared pointer to a view tensor, or nullptr if invalid
         *
         * @note The view borrows data from the parent tensor. The parent must outlive the view.
         * @note Total elements in new_shape must not exceed available elements from offset.
         */
        virtual std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) = 0;
    };

    // Implementation: FP32Tensor.cpp
    /**
     * @brief 32-bit floating point tensor (baseline activation format)
     *
     * **Interfaces implemented**:
     * - `TensorBase`: Core tensor API (inherited)
     * - `IActivationTensor`: Kernel creation, in-place ops (RMSNorm, RoPE, SwiGLU)
     * - `ITensorGemmTileDataProvider`: Block-wise FP32 decode (block_size=32, identity decode)
     * - `IQ8_0Decodable`: On-the-fly quantization to Q8_0 (for testing/compatibility)
     * - `IQ8_1Decodable`: On-the-fly quantization to Q8_1 (for INT8 GEMM activations)
     *
     * **Usage**: Default activation tensor for FP32 inference pipelines. Provides baseline
     * numerical accuracy for parity testing against PyTorch/llama.cpp.
     */
    class FP32Tensor : public TensorBase, public IActivationTensor, public ITensorGemmTileDataProvider, public IQ8_0Decodable, public IQ8_1Decodable
    {
    public:
        explicit FP32Tensor(const std::vector<size_t> &shape, int device_idx = -1);
        ~FP32Tensor() override;

        // ===== TensorBase Interface =====
        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::FP32; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override;

        std::unique_ptr<ITensorGemm> createGemm() override;

        // ===== IActivationTensor Interface =====
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;
        std::unique_ptr<ITensorAttention> createAttention() override;
        ActivationPack to_int8_activation_pack(int rows, int cols) const override;

        bool applyRMSNorm(
            const float *gamma,
            int seq_len,
            int d_model,
            float eps = 1e-6f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        bool applyRoPE(
            float *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta = 10000.0f,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        bool from_int32_with_scales(
            const int32_t *accum,
            int rows,
            int cols,
            const float *row_scales,
            const float *col_scales,
            const float *bias = nullptr) override;

        // Format conversion
        void to_fp32(float *dst) const override;
        void to_bf16(uint16_t *dst) const override;
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override;
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;

        // ===== TensorBase View Support =====
        bool is_view() const override { return is_view_; }
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        // ===== ITensorGemmTileDataProvider Interface =====
        // (Used by QuantizedGemmKernel and auto-tuner for uniform tile access)
        static constexpr size_t FP32_BLOCK_SIZE = 32; // Match quantized formats for fair comparison

        /**
         * @brief Decode block to FP32 (identity operation for FP32Tensor)
         *
         * For FP32Tensor, this is simply a memcpy since data is already in FP32 format.
         * Allows FP32Tensor to work with the same QuantizedGemmKernel infrastructure
         * used by quantized formats (IQ4_NL, Q6_K, etc.).
         */
        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            // Calculate starting position for this block
            const size_t cols = shape_[1];
            const size_t k_start = k_block_offset * FP32_BLOCK_SIZE;
            const float *fp32_row = data() + row_idx * cols + k_start;

            // Determine how many elements to copy (may be less than BLOCK_SIZE at row end)
            const size_t elements_remaining = cols - k_start;
            const size_t elements_to_copy = std::min(static_cast<size_t>(FP32_BLOCK_SIZE), elements_remaining);

            // Copy block data (contiguous memory access)
            std::memcpy(output, fp32_row, elements_to_copy * sizeof(float));

            // Zero-pad remainder if K is not multiple of BLOCK_SIZE
            if (elements_to_copy < FP32_BLOCK_SIZE)
            {
                std::memset(output + elements_to_copy, 0, (FP32_BLOCK_SIZE - elements_to_copy) * sizeof(float));
            }
        }

        const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            // Return pointer to raw FP32 data for this block
            const size_t cols = shape_[1];
            const size_t k_start = k_block_offset * FP32_BLOCK_SIZE;
            return data() + row_idx * cols + k_start;
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return FP32_BLOCK_SIZE; }

        // ===== IQ8_0Decodable Interface =====
        void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const override;

        // ===== IQ8_1Decodable Interface =====
        const Q8_1Block *decode_to_q8_1(size_t row_idx, size_t k_block_offset) const override;

    private:
        // Private constructor for creating views
        FP32Tensor(const std::vector<size_t> &shape,
                   int device_idx,
                   AlignedVector<float> *parent_data,
                   size_t data_offset,
                   std::shared_ptr<FP32Tensor> parent);
        std::vector<size_t> shape_;
        int device_idx_; // -1 = host, ≥0 = device index

        // Ownership model:
        // - If is_view_ == false: owns host_data_ (64-byte aligned for SIMD)
        // - If is_view_ == true: parent_data_ptr_ points to parent's host_data_
        bool is_view_;
        AlignedVector<float> host_data_;        // Owned data (64-byte aligned, only used when !is_view_)
        AlignedVector<float> *parent_data_ptr_; // Borrowed data pointer (only used when is_view_)
        size_t view_offset_;                    // Offset into parent data (only used when is_view_)
        std::shared_ptr<FP32Tensor> parent_;    // Keep parent alive (only used when is_view_)
                                                // Always allocated
        void *device_data_;                     // Allocated if device_idx ≥ 0

        bool host_dirty_;   // Host modified, needs upload
        bool device_dirty_; // Device modified, needs download

        bool sync_to_device();
        bool sync_from_device();
    };

    // Implementation: FP16Tensor.cpp
    /**
     * @brief IEEE 754 half-precision (16-bit) floating point tensor
     *
     * **Format**: 1 sign bit, 5 exponent bits, 10 mantissa bits
     * **Range**: ±65,504 (narrower than FP32, risk of overflow)
     * **Precision**: ~3-4 decimal digits
     * **Memory**: 2× reduction vs FP32
     *
     * **Interfaces implemented**:
     * - `TensorBase`: Core tensor API (inherited)
     * - `IActivationTensor`: Kernel creation, in-place ops (RMSNorm, RoPE, SwiGLU)
     * - `ITensorGemmTileDataProvider`: Block-wise decode to FP32 (block_size=32)
     * - `IQ8_0Decodable`: On-the-fly quantization to Q8_0
     * - `IQ8_1Decodable`: On-the-fly quantization to Q8_1 (for INT8 GEMM activations)
     *
     * **Usage**: Activation tensor for FP16 inference (2× memory savings, GPU-optimized).
     */
    class FP16Tensor : public TensorBase, public IActivationTensor, public ITensorGemmTileDataProvider, public IQ8_0Decodable, public IQ8_1Decodable
    {
    public:
        explicit FP16Tensor(const std::vector<size_t> &shape);
        FP16Tensor(const std::vector<size_t> &shape, const std::vector<uint16_t> &fp16_data);
        ~FP16Tensor() override;

        // TensorBase interface
        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::FP16; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override; // Dequantizes to cache
        float *mutable_data() override;     // Not supported

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // IActivationTensor interface - activation-only operations
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;
        std::unique_ptr<ITensorAttention> createAttention() override;
        ActivationPack to_int8_activation_pack(int rows, int cols) const override;

        bool applyRMSNorm(
            const float *gamma,
            int seq_len,
            int d_model,
            float eps = 1e-6f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        bool applyRoPE(
            float *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta = 10000.0f,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        bool from_int32_with_scales(
            const int32_t *accum,
            int rows,
            int cols,
            const float *row_scales,
            const float *col_scales,
            const float *bias = nullptr) override;

        // Format conversion (TensorBase interface)
        void to_fp32(float *dst) const override;
        void to_bf16(uint16_t *dst) const override;
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override;
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;

        // View support
        bool is_view() const override { return is_view_; }
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        // ITensorGemmTileDataProvider interface - for GEMM kernels to decode FP16→FP32 on-the-fly
        static constexpr size_t FP16_BLOCK_SIZE = 32; // Match quantized formats for cache efficiency

        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            // Calculate starting position for this block
            const size_t cols = shape_[1];
            const size_t k_start = k_block_offset * FP16_BLOCK_SIZE;
            const uint16_t *fp16_row = fp16_data() + row_idx * cols + k_start;

            // Determine how many elements to convert (may be less than BLOCK_SIZE at row end)
            const size_t elements_remaining = cols - k_start;
            const size_t elements_to_convert = std::min(static_cast<size_t>(FP16_BLOCK_SIZE), elements_remaining);

            // Convert block from FP16 to FP32
            for (size_t i = 0; i < elements_to_convert; ++i)
            {
                output[i] = simd::fp16_to_fp32(fp16_row[i]);
            }

            // Zero-pad remainder if K is not multiple of BLOCK_SIZE
            if (elements_to_convert < FP16_BLOCK_SIZE)
            {
                std::memset(output + elements_to_convert, 0, (FP16_BLOCK_SIZE - elements_to_convert) * sizeof(float));
            }
        }

        const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            // Return pointer to raw FP16 data for this block
            const size_t cols = shape_[1];
            const size_t k_start = k_block_offset * FP16_BLOCK_SIZE;
            return fp16_data() + row_idx * cols + k_start;
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return FP16_BLOCK_SIZE; }

        // IQ8_0Decodable interface
        void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const override;

        // IQ8_1Decodable interface
        const Q8_1Block *decode_to_q8_1(size_t row_idx, size_t k_block_offset) const override;

        // FP16-specific interface
        const uint16_t *fp16_data() const
        {
            return is_view_ ? (parent_data_ptr_->data() + view_offset_) : host_fp16_data_.data();
        }
        uint16_t *mutable_fp16_data()
        {
            return is_view_ ? (parent_data_ptr_->data() + view_offset_) : host_fp16_data_.data();
        }
        void from_fp32(const float *fp32_data, size_t count);
        void to_fp32(float *fp32_data, size_t count) const;

    private:
        // Private constructor for creating views
        FP16Tensor(const std::vector<size_t> &shape,
                   int device_idx,
                   AlignedVector<uint16_t> *parent_data,
                   size_t data_offset,
                   std::shared_ptr<FP16Tensor> parent);

        std::vector<size_t> shape_;
        int device_idx_;

        // Ownership model:
        // - If is_view_ == false: owns host_fp16_data_ (64-byte aligned for SIMD)
        // - If is_view_ == true: parent_data_ptr_ points to parent's host_fp16_data_
        bool is_view_;
        AlignedVector<uint16_t> host_fp16_data_;   // Owned data (64-byte aligned, only used when !is_view_)
        AlignedVector<uint16_t> *parent_data_ptr_; // Borrowed data pointer (only used when is_view_)
        size_t view_offset_;                       // Offset into parent data (only used when is_view_)
        std::shared_ptr<FP16Tensor> parent_;       // Keep parent alive (only used when is_view_)

        void *device_data_;                          // Device-side storage
        mutable AlignedVector<float> dequant_cache_; // For data() calls (64-byte aligned)

        bool sync_to_device();
        bool sync_from_device();
    };

    // Implementation: BF16Tensor.cpp
    /**
     * @brief Brain Float 16 tensor (Google's reduced-precision format)
     *
     * **Format**: 1 sign bit, 8 exponent bits, 7 mantissa bits
     * **Range**: Same as FP32 (prevents overflow/underflow issues)
     * **Precision**: ~2-3 decimal digits (lower than FP16)
     * **Memory**: 2× reduction vs FP32
     * **Hardware**: Accelerated on Intel Ice Lake+, AMD Zen 4+, NVIDIA Ampere+
     *
     * **Interfaces implemented**:
     * - `TensorBase`: Core tensor API (inherited)
     * - `IActivationTensor`: Kernel creation, in-place ops (RMSNorm, RoPE, SwiGLU)
     * - `ITensorGemmTileDataProvider`: Block-wise decode to FP32 (block_size=32)
     * - `IQ8_0Decodable`: On-the-fly quantization to Q8_0
     * - `IQ8_1Decodable`: On-the-fly quantization to Q8_1 (for INT8 GEMM activations)
     *
     * **Usage**: Preferred activation tensor for CPU/GPU mixed precision (better overflow resistance than FP16).
     */
    class BF16Tensor : public TensorBase, public IActivationTensor, public ITensorGemmTileDataProvider, public IQ8_0Decodable, public IQ8_1Decodable
    {
    public:
        explicit BF16Tensor(const std::vector<size_t> &shape);
        BF16Tensor(const std::vector<size_t> &shape, const std::vector<uint16_t> &bf16_data);
        ~BF16Tensor() override;

        // TensorBase interface
        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::BF16; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override; // Dequantizes to cache
        float *mutable_data() override;     // Not supported

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // IActivationTensor interface - activation-only operations
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;
        std::unique_ptr<ITensorAttention> createAttention() override;
        ActivationPack to_int8_activation_pack(int rows, int cols) const override;

        bool applyRMSNorm(
            const float *gamma,
            int seq_len,
            int d_model,
            float eps = 1e-6f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        bool applyRoPE(
            float *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta = 10000.0f,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        bool from_int32_with_scales(
            const int32_t *accum,
            int rows,
            int cols,
            const float *row_scales,
            const float *col_scales,
            const float *bias = nullptr) override;

        // Format conversion (TensorBase interface)
        void to_fp32(float *dst) const override;
        void to_bf16(uint16_t *dst) const override;
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override;
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;

        // View support
        bool is_view() const override { return is_view_; }
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        // ITensorGemmTileDataProvider interface - for GEMM kernels to decode BF16→FP32 on-the-fly
        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            // BF16 tensors are NOT block-quantized - they're element-wise BF16 values
            // For compatibility with micro-kernel template, treat each row as contiguous "blocks"
            // where block_size = number of columns
            const size_t cols = shape_[1];
            const uint16_t *bf16_row = bf16_data() + row_idx * cols;

            // Convert entire row from BF16 to FP32
            for (size_t i = 0; i < cols; ++i)
            {
                output[i] = simd::bf16_to_fp32(bf16_row[i]);
            }
        }

        const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            // Return pointer to raw BF16 data for this row
            const size_t cols = shape_[1];
            return bf16_data() + row_idx * cols;
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return shape_[1]; } // Entire row is one "block"

        // BF16-specific interface (deprecated - use TensorBase methods instead)
        const uint16_t *bf16_data() const
        {
            return is_view_ ? (parent_data_ptr_->data() + view_offset_) : host_bf16_data_.data();
        }
        uint16_t *mutable_bf16_data()
        {
            return is_view_ ? (parent_data_ptr_->data() + view_offset_) : host_bf16_data_.data();
        }
        void from_fp32(const float *fp32_data, size_t count);
        void to_fp32(float *fp32_data, size_t count) const; // Deprecated overload

        // IQ8_0Decodable interface
        void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const override;

        // IQ8_1Decodable interface - returns pointer to thread-local block
        const Q8_1Block *decode_to_q8_1(size_t row_idx, size_t k_block_offset) const override;

    private:
        // Private constructor for creating views
        BF16Tensor(const std::vector<size_t> &shape,
                   int device_idx,
                   AlignedVector<uint16_t> *parent_data,
                   size_t data_offset,
                   std::shared_ptr<BF16Tensor> parent);

        std::vector<size_t> shape_;
        int device_idx_;

        // Ownership model:
        // - If is_view_ == false: owns host_bf16_data_ (64-byte aligned for SIMD)
        // - If is_view_ == true: parent_data_ptr_ points to parent's host_bf16_data_
        bool is_view_;
        AlignedVector<uint16_t> host_bf16_data_;   // Owned data (64-byte aligned, only used when !is_view_)
        AlignedVector<uint16_t> *parent_data_ptr_; // Borrowed data pointer (only used when is_view_)
        size_t view_offset_;                       // Offset into parent data (only used when is_view_)
        std::shared_ptr<BF16Tensor> parent_;       // Keep parent alive (only used when is_view_)

        void *device_data_;                          // Device-side storage
        mutable AlignedVector<float> dequant_cache_; // For data() calls (64-byte aligned)

        bool sync_to_device();
        bool sync_from_device();
    };

    // Forward declare for ITensorGemmTileDataProvider
    class ITensorGemmTileDataProvider;

    // Implementation: INT8Tensor.cpp
    /**
     * @brief 8-bit signed integer tensor for quantized GEMM (AVX512-VNNI, CUDA INT8 Tensor Cores)
     *
     * **Purpose**: Dequantized weight storage for INT8×INT8 GEMM operations.
     * When `--weight-precision int8` is set, all quantized weight tensors (IQ4_NL, Q6_K, Q8_0, etc.)
     * are converted to INT8Tensor at model load time.
     *
     * **Storage**: 8-bit signed integers + per-column and per-row scaling factors
     * **Memory**: 4× compression vs FP32 (8 bits vs 32 bits per element)
     *
     * **Hardware Support**:
     * - CPU: AVX512-VNNI (Ice Lake+), AVX2-VNNI (Alder Lake+), AMX (Sapphire Rapids+)
     * - GPU: CUDA INT8 Tensor Cores (Turing+), cuBLAS INT8 GEMM
     *
     * **Interfaces implemented**:
     * - `TensorBase`: Core tensor API (inherited)
     * - `ITensorGemmTileDataProvider`: Block-wise decode to FP32 for mixed-precision GEMM
     *
     * **NOT implemented**: IActivationTensor (INT8Tensor represents weights, not activations)
     */
    class INT8Tensor : public TensorBase, public ITensorGemmTileDataProvider
    {
    public:
        explicit INT8Tensor(const std::vector<size_t> &shape);
        INT8Tensor(const std::vector<size_t> &shape,
                   const std::vector<int8_t> &data,
                   float scale);
        INT8Tensor(const std::vector<size_t> &shape,
                   const std::vector<float> &fp32_data);
        ~INT8Tensor() override = default;

        // TensorBase interface
        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::INT8; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override; // Dequantizes to cache
        float *mutable_data() override;     // Not supported

        bool copyFrom(const TensorBase *src) override;

        std::unique_ptr<ITensorGemm> createGemm() override;

        bool from_int32_with_scales(
            const int32_t *accum,
            int rows,
            int cols,
            const float *row_scales,
            const float *col_scales,
            const float *bias = nullptr);

        // Format conversion
        void to_fp32(float *dst) const override;
        void to_bf16(uint16_t *dst) const override;
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override;
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;

        // View support (future)
        bool is_view() const override { return false; }
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        // INT8-specific interface
        const int8_t *int8_data() const { return host_int8_data_.data(); }
        int8_t *mutable_int8_data() { return host_int8_data_.data(); }
        float scale() const { return scale_; }
        void set_scale(float s) { scale_ = s; }

        // Per-column scales (for weight matrices)
        const float *col_scales() const { return col_scales_.empty() ? nullptr : col_scales_.data(); }
        bool has_col_scales() const { return !col_scales_.empty(); }
        size_t num_col_scales() const { return col_scales_.size(); }
        void set_col_scales(const std::vector<float> &scales) { col_scales_ = scales; }
        void set_col_scales(const float *scales, size_t count)
        {
            col_scales_.assign(scales, scales + count);
        }

        // Per-row scales (computed on-demand for transpose operations)
        const std::vector<float> &get_row_scales() const;
        bool has_row_scales() const { return !row_scales_cache_.empty(); }
        void set_row_scales(const std::vector<float> &scales) { row_scales_cache_ = scales; }
        void set_row_scales(const float *scales, size_t count)
        {
            row_scales_cache_.assign(scales, scales + count);
        }

        // ITensorGemmTileDataProvider interface - for INT8 GEMM kernels
        void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override;

        const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override;

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return shape_[1]; } // Full row per block

    private:
        std::vector<size_t> shape_;
        int device_idx_ = -1;
        AlignedVector<int8_t> host_int8_data_;        // 64-byte aligned for SIMD operations
        float scale_ = 1.0f;                          ///< Global scale factor (fallback if col_scales_ empty)
        std::vector<float> col_scales_;               ///< Per-column scales (for 2D weight matrices)
        mutable std::vector<float> row_scales_cache_; ///< Cached per-row scales (computed on-demand)
        void *device_data_ = nullptr;
        mutable AlignedVector<float> dequant_cache_; // 64-byte aligned dequant buffer

        bool sync_to_device();
        bool sync_from_device();
    };

    // Implementation: INT32Tensor.cpp
    /**
     * @brief 32-bit signed integer accumulator tensor for full INT8 inference pipeline
     *
     * **Purpose**: Stores INT32 accumulator results from INT8×INT8 GEMM operations.
     * Enables full INT8 inference by supporting requantization back to INT8 activations
     * for the next layer (avoiding expensive INT32→FP32→INT8 round-trip).
     *
     * **Pipeline Flow**:
     * 1. INT8 activation × INT8 weight → INT32 accumulator (VNNI/Tensor Core output)
     * 2. INT32Tensor stores result with per-row scaling metadata
     * 3. `requantize_to_int8()` → INT8 activation for next layer
     * 4. Final layer: `to_fp32()` for logits output
     *
     * **Interfaces implemented**:
     * - `TensorBase`: Core tensor API (inherited)
     * - `IActivationTensor`: Kernel creation, format conversions (treated as activation, not weight)
     *
     * **Key methods**:
     * - `requantize_to_int8()`: Dynamic per-row rescaling for next layer
     * - `from_int32_with_scales()`: Populate from raw INT32 accumulator + scale factors
     * - `to_fp32()`: Dequantize to FP32 for final output or parity testing
     */
    class INT32Tensor : public TensorBase
    {
    public:
        explicit INT32Tensor(const std::vector<size_t> &shape);
        INT32Tensor(const std::vector<size_t> &shape,
                    const std::vector<int32_t> &data);
        INT32Tensor(const std::vector<size_t> &shape,
                    const std::vector<float> &fp32_data,
                    float scale);
        ~INT32Tensor() override = default;

        // TensorBase interface
        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::INT32; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override; // Dequantizes to cache
        float *mutable_data() override;     // Not supported

        bool copyFrom(const TensorBase *src) override;

        std::unique_ptr<ITensorGemm> createGemm() override;

        bool from_int32_with_scales(
            const int32_t *accum,
            int rows,
            int cols,
            const float *row_scales,
            const float *col_scales,
            const float *bias = nullptr);

        // Format conversion
        void to_fp32(float *dst) const override;
        void to_bf16(uint16_t *dst) const override;
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override;
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;

        // View support (not yet implemented)
        bool is_view() const override { return false; }
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        // INT32-specific interface
        const int32_t *int32_data() const { return host_int32_data_.data(); }
        int32_t *mutable_int32_data() { return host_int32_data_.data(); }
        float scale() const { return scale_; }
        void set_scale(float s) { scale_ = s; }

        // Per-row scales (for INT32→INT8 requantization)
        const float *row_scales() const { return row_scales_.empty() ? nullptr : row_scales_.data(); }
        bool has_row_scales() const { return !row_scales_.empty(); }
        size_t num_row_scales() const { return row_scales_.size(); }
        void set_row_scales(const std::vector<float> &scales) { row_scales_ = scales; }

        /**
         * @brief Requantize INT32 to INT8 with per-row dynamic scaling
         *
         * This is the KEY function for full INT8 pipelines. Converts INT32
         * accumulator results back to INT8 for the next layer.
         *
         * Per-row quantization maintains better accuracy than per-tensor:
         * - Each row has independent dynamic range
         * - Prevents outliers in one row from reducing precision in others
         *
         * @param dst_int8 Output INT8 data [m, n]
         * @param dst_row_scales Output per-row scales [m]
         * @return True if successful
         */
        bool requantize_to_int8(int8_t *dst_int8, float *dst_row_scales) const;

    private:
        std::vector<size_t> shape_;
        int device_idx_ = -1;
        AlignedVector<int32_t> host_int32_data_; // 64-byte aligned for SIMD operations
        float scale_ = 1.0f;                     ///< Global scale factor
        std::vector<float> row_scales_;          ///< Per-row scales (optional)
        void *device_data_ = nullptr;
        mutable AlignedVector<float> dequant_cache_; ///< Cached FP32 dequantization (64-byte aligned)

        bool sync_to_device();
        bool sync_from_device();
    };

    // Implementation: IQ4_NLTensor.cpp
    /**
     * @brief IQ4_NL (4-bit non-linear) quantized weight tensor
     *
     * **Format**: 4.5 bits per weight (4-bit quantized values + FP16 scale)
     * **Compression**: 7.1× vs FP32 (best quality-to-size ratio for 4-bit)
     * **Quantization**: Non-linear lookup table optimized for LLM weight distributions
     *
     * **Block Structure** (32 elements per block):
     * - 16 bytes: Packed 4-bit quantized values (2 values per byte)
     * - 2 bytes: FP16 scale factor
     * - Total: 18 bytes per 32 weights → 4.5 bits/weight
     *
     * **Interfaces implemented**:
     * - `TensorBase`: Core tensor API (inherited)
     * - `ITensorGemmTileDataProvider`: Block-wise decode to FP32 (for CPU FP32 GEMM)
     * - `IQ8_0Decodable`: On-the-fly conversion to Q8_0 (for INT8 GEMM weight caching)
     *
     * **NOT implemented**: IActivationTensor (IQ4_NL is read-only weight format)
     *
     * **Usage**: Default quantization format for model weights (balance of size and quality).
     */

    class IQ4_NLTensor : public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable
    {
    public:
        IQ4_NLTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~IQ4_NLTensor() override;

        // TensorBase interface
        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ4_NL; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override; // Dequantizes to temp buffer
        float *mutable_data() override;     // Not supported for quantized tensors

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override; // Fused dequant+GEMM

        // Format conversion (TensorBase interface - delegates to decode methods)
        void to_fp32(float *dst) const override { decode_to_fp32(dst); }
        void to_bf16(uint16_t *dst) const override { decode_to_bf16(dst); }
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override;

        /**
         * @brief Convert tensor rows to INT8 with per-row scales (avoids FP32 round-trip)
         * @param dst_int8 Destination buffer sized rows * cols (row-major)
         * @param dst_row_scales Destination buffer sized rows (per-row scales)
         * @return true on success, false otherwise
         */
        bool to_int8_rowmajor(int8_t *dst_int8, float *dst_row_scales) const;
        void to_fp32_row(size_t row_idx, float *buffer) const override { decodeRow(row_idx, buffer); }
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override { decodeSpan(offset, count, buffer); }

        // View support (row-slice only - preserves K dimension)
        bool is_view() const override { return is_view_; }
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        // Shape and metadata
        size_t size() const { return shape_[0] * shape_[1]; }
        size_t ndim() const { return 2; }
        float compression_ratio() const { return 7.1f; }
        size_t logical_k() const { return shape_[1]; }
        size_t padded_k() const;
        size_t element_count() const { return shape_[0] * shape_[1]; }

        // Raw data access
        const uint8_t *raw_data() const { return is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data(); }
        const uint8_t *raw_blocks() const { return is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data(); }
        size_t raw_size() const
        {
            if (is_view_)
            {
                size_t blocks_per_row = (shape_[1] + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE;
                return shape_[0] * blocks_per_row * sizeof(IQ4_NLBlock);
            }
            return raw_data_.size();
        }
        size_t num_blocks() const { return raw_size() / sizeof(IQ4_NLBlock); }

        // Decode API
        void decode_to_fp32(float *dst) const;
        void decode_to_bf16(uint16_t *dst) const;
        void decodeRow(size_t row_idx, float *buffer) const;
        void decodeSpan(size_t offset, size_t count, float *buffer) const;
        void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const;

        // Fused kernel helpers (non-virtual versions for backward compatibility)
        const IQ4_NLBlock &get_block_at(size_t row_idx, size_t k_block_offset) const;
        void decode_tile_blocks(size_t row_start, size_t tile_n, size_t k_block_offset, float *output) const;

        // ITensorGemmTileDataProvider interface (inline for zero overhead in GEMM hot path)
        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            const size_t blocks_per_row = (shape_[1] + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock *>(data_ptr);
            const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
            decodeBlock(blocks[block_idx], output);
        }

        __attribute__((always_inline))
        const void *
        get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t blocks_per_row = (shape_[1] + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock *>(data_ptr);
            const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
            return &blocks[block_idx];
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return IQ4_NLBlock::BLOCK_SIZE; }

        // SIMD decode methods (public for testing)
        static void decodeBlock(const IQ4_NLBlock &block, float *output);
        static void decodeBlockScalar(const IQ4_NLBlock &block, float *output);
#ifdef __AVX512F__
        static void decodeBlockAVX512(const IQ4_NLBlock &block, float *output);
        static inline void decodeBlockVectorizedAVX512(const IQ4_NLBlock &block, float *output);
#endif
#ifdef __AVX2__
        static void decodeBlockAVX2(const IQ4_NLBlock &block, float *output);
        static void decodeBlockVectorizedAVX2(const IQ4_NLBlock &block, float *output);
#endif

    private:
        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)

        int device_idx_;
        void *device_blocks_; // Quantized blocks on device (if uploaded)

        mutable std::vector<float> dequant_cache_; // Temporary for data() calls

        // Private view constructor
        IQ4_NLTensor(const std::vector<size_t> &shape,
                     const uint8_t *parent_raw_data,
                     size_t byte_offset,
                     std::shared_ptr<TensorBase> parent);

        // Decode helpers
        static void decode_to_fp32_microkernel(float *dst, const IQ4_NLBlock *blocks, int rows, int cols, size_t blocks_per_row);
    };

    // ===== Q8_0 Tensor (8-bit quantization) =====

    // Implementation: Q8_0Tensor.cpp
    /**
     * @brief Q8_0 quantized tensor (8-bit uniform quantization)
     *
     * Block format: 32 elements per block, FP16 scale + int8[32] values
     * Compression: 4× vs FP32
     */
    class Q8_0Tensor : public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable
    {
    public:
        Q8_0Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~Q8_0Tensor() override;

        // TensorBase interface
        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q8_0; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Format conversion (TensorBase interface)
        void to_fp32(float *dst) const override { to_fp32_via_blocks(dst); }
        void to_bf16(uint16_t *dst) const override;
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override;
        bool to_int8_rowmajor(int8_t *dst_int8, float *dst_row_scales) const;
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;

        // IQ8_0Decodable interface - Q8_0 to Q8_0 is a direct copy (no conversion needed)
        void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const override;

        // View support (row-slice only - preserves K dimension)
        bool is_view() const override { return is_view_; }
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        // ITensorGemmTileDataProvider interface (inline for zero overhead in GEMM hot path)
        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q8_0Block *blocks = reinterpret_cast<const Q8_0Block *>(data_ptr);
            const Q8_0Block &block = blocks[row_idx * blocks_per_row + k_block_offset];
            decodeBlock(block, output);
        }

        __attribute__((always_inline))
        const void *
        get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q8_0Block *blocks = reinterpret_cast<const Q8_0Block *>(data_ptr);
            return &blocks[row_idx * blocks_per_row + k_block_offset];
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return Q8_0Block::BLOCK_SIZE; }

        // Public decode methods for testing
        static void decodeBlock(const Q8_0Block &block, float *output);
        static void decodeBlockScalar(const Q8_0Block &block, float *output);

#if defined(__AVX512F__)
        static void decodeBlockAVX512(const Q8_0Block &block, float *output);
#endif

#if defined(__AVX2__)
        static void decodeBlockAVX2(const Q8_0Block &block, float *output);
#endif

    private:
        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)

        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;

        // Private view constructor
        Q8_0Tensor(const std::vector<size_t> &shape,
                   const uint8_t *parent_raw_data,
                   size_t byte_offset,
                   std::shared_ptr<TensorBase> parent);
    };

    // ===== Q8_1 Tensor (8-bit quantization with pre-computed sum) =====

    // Implementation: Q8_1Tensor.cpp
    /**
     * @brief Q8_1 quantized tensor (8-bit uniform quantization with pre-computed sum)
     *
     * Block format: 32 elements per block, FP16 scale + FP16 sum + int8[32] values
     * Compression: 4× vs FP32 (slight overhead vs Q8_0 for pre-computed sum)
     *
     * This format is used as an intermediate activation format (matching CUDA pattern):
     * - Pre-computes sum during quantization: s = d × sum(qs[i])
     * - Eliminates expensive horizontal reductions in GEMM K-loop
     * - "Quantize once, use many times" amortization
     *
     * Typical usage:
     *   FP32/FP16/BF16 activations → Q8_1 (quantize once per panel)
     *   → Q8_1 activations × quantized weights (many dot products, no sum computation!)
     */
    class Q8_1Tensor : public TensorBase, public IActivationTensor, public ITensorGemmTileDataProvider, public IQ8_1Decodable
    {
    public:
        using value_type = Q8_1Block;

        Q8_1Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~Q8_1Tensor() override;

        // TensorBase interface
        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q8_1; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override;

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Format conversion (TensorBase interface)
        void to_fp32(float *dst) const override { to_fp32_via_blocks(dst); }
        void to_bf16(uint16_t *dst) const override;
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override
        {
            return to_int8_perchannel_via_blocks(dst_int8, dst_col_scales, dst_row_scales);
        }
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;

        // IQ8_1Decodable interface - Q8_1 to Q8_1 returns direct pointer (zero-cost)
        const Q8_1Block *decode_to_q8_1(size_t row_idx, size_t k_block_offset) const override;

        // IActivationTensor interface (Q8_1 activations)
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;
        std::unique_ptr<ITensorAttention> createAttention() override;
        ActivationPack to_int8_activation_pack(int rows, int cols) const override;

        bool applyRMSNorm(
            const float *gamma,
            int seq_len,
            int d_model,
            float eps = 1e-6f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        bool applyRoPE(
            float *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta = 10000.0f,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        bool from_int32_with_scales(
            const int32_t *accum,
            int rows,
            int cols,
            const float *row_scales,
            const float *col_scales,
            const float *bias = nullptr) override;

        // View support (row-slice only - preserves K dimension)
        bool is_view() const override { return is_view_; }
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        // ITensorGemmTileDataProvider interface (inline for zero overhead in GEMM hot path)
        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q8_1Block *blocks = reinterpret_cast<const Q8_1Block *>(data_ptr);
            const Q8_1Block &block = blocks[row_idx * blocks_per_row + k_block_offset];
            decodeBlock(block, output);
        }

        __attribute__((always_inline))
        const void *
        get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q8_1Block *blocks = reinterpret_cast<const Q8_1Block *>(data_ptr);
            return &blocks[row_idx * blocks_per_row + k_block_offset];
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return Q8_1Block::BLOCK_SIZE; }

        // Public decode methods for testing
        static void decodeBlock(const Q8_1Block &block, float *output);
        static void decodeBlockScalar(const Q8_1Block &block, float *output);

#if defined(__AVX512F__)
        static void decodeBlockAVX512(const Q8_1Block &block, float *output);
#endif

#if defined(__AVX2__)
        static void decodeBlockAVX2(const Q8_1Block &block, float *output);
#endif

        /**
         * @brief Quantize FP32 data to Q8_1 format with pre-computed sum
         *
         * This is the key method that implements the CUDA-style "quantize once, use many times" pattern.
         * It computes the sum during quantization (essentially free since we're touching the data anyway).
         *
         * @param src Source FP32 data
         * @param shape Tensor shape [rows, cols]
         * @return Q8_1Tensor with pre-computed sums
         */
        static std::shared_ptr<Q8_1Tensor> quantize_from_fp32(
            const float *src,
            const std::vector<size_t> &shape);

    private:
        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)

        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;

        // Private view constructor
        Q8_1Tensor(const std::vector<size_t> &shape,
                   const uint8_t *parent_raw_data,
                   size_t byte_offset,
                   std::shared_ptr<TensorBase> parent);
    };

    // ===== Q4_0 Tensor (4-bit quantization) =====

    // Implementation: Q4_0Tensor.cpp
    /**
     * @brief Q4_0 quantized tensor (4-bit uniform quantization)
     *
     * Block format: 32 elements per block, FP16 scale + 4-bit packed values
     * Compression: 8× vs FP32
     */
    class Q4_0Tensor : public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable
    {
    public:
        Q4_0Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~Q4_0Tensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q4_0; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Format conversion (TensorBase interface)
        void to_fp32(float *dst) const override { to_fp32_via_blocks(dst); }
        void to_bf16(uint16_t *dst) const override;
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override
        {
            return to_int8_perchannel_via_blocks(dst_int8, dst_col_scales, dst_row_scales);
        }
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;

        // Per-block decode to Q8_0 (used by Q8_0WeightAccessor)
        void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const override;

        // View support (row-slice only due to block alignment)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q4_0Block::BLOCK_SIZE - 1) / Q4_0Block::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q4_0Block *blocks = reinterpret_cast<const Q4_0Block *>(data_ptr);
            decodeBlock(blocks[row_idx * blocks_per_row + k_block_offset], output);
        }

        __attribute__((always_inline))
        const void *
        get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q4_0Block::BLOCK_SIZE - 1) / Q4_0Block::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q4_0Block *blocks = reinterpret_cast<const Q4_0Block *>(data_ptr);
            return &blocks[row_idx * blocks_per_row + k_block_offset];
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return Q4_0Block::BLOCK_SIZE; }

        // SIMD decode methods (public for unit testing)
        static void decodeBlock(const Q4_0Block &block, float *output);
        static void decodeBlockScalar(const Q4_0Block &block, float *output);

#if defined(__AVX512F__)
        static void decodeBlockAVX512(const Q4_0Block &block, float *output);
#endif

#if defined(__AVX2__)
        static void decodeBlockAVX2(const Q4_0Block &block, float *output);
#endif

    private:
        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)

        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;

        // Private view constructor
        Q4_0Tensor(const std::vector<size_t> &shape,
                   const uint8_t *parent_raw_data,
                   size_t byte_offset,
                   std::shared_ptr<TensorBase> parent);
    };

    // ===== Q4_1 Tensor (4-bit with min) =====

    // Implementation: Q4_1Tensor.cpp
    /**
     * @brief Q4_1 quantized tensor (4-bit with min offset)
     *
     * Block format: 32 elements per block, FP16 scale + FP16 min + 4-bit packed values
     * Compression: ~7.1× vs FP32
     */
    class Q4_1Tensor : public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable
    {
    public:
        Q4_1Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~Q4_1Tensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q4_1; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Format conversion (TensorBase interface)
        void to_fp32(float *dst) const override { to_fp32_via_blocks(dst); }
        void to_bf16(uint16_t *dst) const override;
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override
        {
            return to_int8_perchannel_via_blocks(dst_int8, dst_col_scales, dst_row_scales);
        }
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;

        // Per-block decode to Q8_0 (used by Q8_0WeightAccessor)
        void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const;

        // View support (row-slice only due to block alignment)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q4_1Block::BLOCK_SIZE - 1) / Q4_1Block::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q4_1Block *blocks = reinterpret_cast<const Q4_1Block *>(data_ptr);
            decodeBlock(blocks[row_idx * blocks_per_row + k_block_offset], output);
        }

        __attribute__((always_inline))
        const void *
        get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q4_1Block::BLOCK_SIZE - 1) / Q4_1Block::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q4_1Block *blocks = reinterpret_cast<const Q4_1Block *>(data_ptr);
            return &blocks[row_idx * blocks_per_row + k_block_offset];
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return Q4_1Block::BLOCK_SIZE; }

        // SIMD decode methods (public for unit testing)
        static void decodeBlock(const Q4_1Block &block, float *output);
        static void decodeBlockScalar(const Q4_1Block &block, float *output);

#if defined(__AVX512F__)
        static void decodeBlockAVX512(const Q4_1Block &block, float *output);
#endif

#if defined(__AVX2__)
        static void decodeBlockAVX2(const Q4_1Block &block, float *output);
#endif

    private:
        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)

        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;

        // Private view constructor
        Q4_1Tensor(const std::vector<size_t> &shape,
                   const uint8_t *parent_raw_data,
                   size_t byte_offset,
                   std::shared_ptr<TensorBase> parent);
    };

    // ===== Q5_0 Tensor (5-bit uniform quantization) =====

    // Implementation: Q5_0Tensor.cpp
    /**
     * @brief Q5_0 quantized tensor (5-bit uniform quantization)
     *
     * Block format: 32 elements per block, FP16 scale + 5-bit packed values
     * High bit stored separately in qh[4] array (32 bits for 32 elements)
     * Compression: ~6.4× vs FP32
     */
    class Q5_0Tensor : public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable
    {
    public:
        Q5_0Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~Q5_0Tensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q5_0; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Format conversion (TensorBase interface)
        void to_fp32(float *dst) const override { to_fp32_via_blocks(dst); }
        void to_bf16(uint16_t *dst) const override;
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override
        {
            return to_int8_perchannel_via_blocks(dst_int8, dst_col_scales, dst_row_scales);
        }
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;

        // Per-block decode to Q8_0 (used by Q8_0WeightAccessor)
        void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const;

        // View support (row-slice only due to block alignment)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q5_0Block::BLOCK_SIZE - 1) / Q5_0Block::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q5_0Block *blocks = reinterpret_cast<const Q5_0Block *>(data_ptr);
            decodeBlock(blocks[row_idx * blocks_per_row + k_block_offset], output);
        }

        __attribute__((always_inline))
        const void *
        get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q5_0Block::BLOCK_SIZE - 1) / Q5_0Block::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q5_0Block *blocks = reinterpret_cast<const Q5_0Block *>(data_ptr);
            return &blocks[row_idx * blocks_per_row + k_block_offset];
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return Q5_0Block::BLOCK_SIZE; }

        // SIMD decode methods (public for unit testing)
        static void decodeBlock(const Q5_0Block &block, float *output);
        static void decodeBlockScalar(const Q5_0Block &block, float *output);

#if defined(__AVX512F__)
        static void decodeBlockAVX512(const Q5_0Block &block, float *output);
#endif

#if defined(__AVX2__)
        static void decodeBlockAVX2(const Q5_0Block &block, float *output);
#endif

    private:
        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)

        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;

        // Private view constructor
        Q5_0Tensor(const std::vector<size_t> &shape,
                   const uint8_t *parent_raw_data,
                   size_t byte_offset,
                   std::shared_ptr<TensorBase> parent);
    };

    // ===== Q5_1 Tensor (5-bit with min) =====

    // Implementation: Q5_1Tensor.cpp
    /**
     * @brief Q5_1 quantized tensor (5-bit with min offset)
     *
     * Block format: 32 elements per block, FP16 scale + FP16 min + 5-bit packed values
     * High bit stored separately in qh[4] array (32 bits for 32 elements)
     * Compression: ~5.7× vs FP32
     */
    class Q5_1Tensor : public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable
    {
    public:
        Q5_1Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~Q5_1Tensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q5_1; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Format conversion (TensorBase interface)
        void to_fp32(float *dst) const override { to_fp32_via_blocks(dst); }
        void to_bf16(uint16_t *dst) const override;
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override
        {
            return to_int8_perchannel_via_blocks(dst_int8, dst_col_scales, dst_row_scales);
        }
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;

        // Per-block decode to Q8_0 (used by Q8_0WeightAccessor)
        void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const;

        // View support (row-slice only due to block alignment)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q5_1Block::BLOCK_SIZE - 1) / Q5_1Block::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q5_1Block *blocks = reinterpret_cast<const Q5_1Block *>(data_ptr);
            decodeBlock(blocks[row_idx * blocks_per_row + k_block_offset], output);
        }

        __attribute__((always_inline))
        const void *
        get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q5_1Block::BLOCK_SIZE - 1) / Q5_1Block::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q5_1Block *blocks = reinterpret_cast<const Q5_1Block *>(data_ptr);
            return &blocks[row_idx * blocks_per_row + k_block_offset];
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return Q5_1Block::BLOCK_SIZE; }

        // SIMD decode methods (public for unit testing)
        static void decodeBlock(const Q5_1Block &block, float *output);
        static void decodeBlockScalar(const Q5_1Block &block, float *output);

#if defined(__AVX512F__)
        static void decodeBlockAVX512(const Q5_1Block &block, float *output);
#endif

#if defined(__AVX2__)
        static void decodeBlockAVX2(const Q5_1Block &block, float *output);
#endif

    private:
        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)

        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;

        // Private view constructor
        Q5_1Tensor(const std::vector<size_t> &shape,
                   const uint8_t *parent_raw_data,
                   size_t byte_offset,
                   std::shared_ptr<TensorBase> parent);
    };

    // ===== K-quant Tensors =====

    // Implementation: Q6_KTensor.cpp
    /**
     * @brief Q6_K tensor (6-bit K-quant super-block)
     */
    class Q6_KTensor : public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable
    {
    public:
        Q6_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~Q6_KTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q6_K; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Format conversion (TensorBase interface)
        void to_fp32(float *dst) const override { to_fp32_via_blocks(dst); }
        void to_bf16(uint16_t *dst) const override;
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override
        {
            return to_int8_perchannel_via_blocks(dst_int8, dst_col_scales, dst_row_scales);
        }
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;

        // View support (row-slice only, preserves 256-element super-block alignment)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q6_KBlock::BLOCK_SIZE - 1) / Q6_KBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q6_KBlock *blocks = reinterpret_cast<const Q6_KBlock *>(data_ptr);
            decodeBlock(blocks[row_idx * blocks_per_row + k_block_offset], output);
        }

        __attribute__((always_inline))
        const void *
        get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q6_KBlock::BLOCK_SIZE - 1) / Q6_KBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q6_KBlock *blocks = reinterpret_cast<const Q6_KBlock *>(data_ptr);
            return &blocks[row_idx * blocks_per_row + k_block_offset];
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return Q6_KBlock::BLOCK_SIZE; }

        // Q8_0 decode (for integer GEMM)
        void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const;

        // SIMD decode methods (public for unit testing)
        static void decodeBlock(const Q6_KBlock &block, float *output);
        static void decodeBlockScalar(const Q6_KBlock &block, float *output);

#if defined(__AVX512F__)
        static void decodeBlockAVX512(const Q6_KBlock &block, float *output);
#endif

#if defined(__AVX2__)
        static void decodeBlockAVX2(const Q6_KBlock &block, float *output);
#endif

    private:
        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)

        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;

        // Private view constructor
        Q6_KTensor(const std::vector<size_t> &shape,
                   const uint8_t *parent_raw_data,
                   size_t byte_offset,
                   std::shared_ptr<TensorBase> parent);
    };

    // Implementation: Q2_KTensor.cpp
    /**
     * @brief Q2_K tensor (2-bit K-quant super-block)
     */
    class Q2_KTensor : public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable
    {
    public:
        Q2_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~Q2_KTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q2_K; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Format conversion (TensorBase interface)
        void to_fp32(float *dst) const override { to_fp32_via_blocks(dst); }
        void to_bf16(uint16_t *dst) const override;
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override
        {
            return to_int8_perchannel_via_blocks(dst_int8, dst_col_scales, dst_row_scales);
        }
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;

        // Per-block decode to Q8_0 (used by Q8_0WeightAccessor)
        void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const;

        // View support (row-slice only, preserves 256-element super-block alignment)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q2_KBlock::BLOCK_SIZE - 1) / Q2_KBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q2_KBlock *blocks = reinterpret_cast<const Q2_KBlock *>(data_ptr);
            decodeBlock(blocks[row_idx * blocks_per_row + k_block_offset], output);
        }

        __attribute__((always_inline))
        const void *
        get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q2_KBlock::BLOCK_SIZE - 1) / Q2_KBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q2_KBlock *blocks = reinterpret_cast<const Q2_KBlock *>(data_ptr);
            return &blocks[row_idx * blocks_per_row + k_block_offset];
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return Q2_KBlock::BLOCK_SIZE; }

        // SIMD decode methods exposed for unit testing
        static void decodeBlockScalar(const Q2_KBlock &block, float *output);

#if defined(__AVX2__)
        static void decodeBlockAVX2(const Q2_KBlock &block, float *output);
#endif

#if defined(__AVX512F__)
        static void decodeBlockAVX512(const Q2_KBlock &block, float *output);
#endif

    private:
        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)

        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;

        // Private view constructor
        Q2_KTensor(const std::vector<size_t> &shape,
                   const uint8_t *parent_raw_data,
                   size_t byte_offset,
                   std::shared_ptr<TensorBase> parent);

        static void decodeBlock(const Q2_KBlock &block, float *output);
    };

    // Implementation: Q5_KTensor.cpp
    /**
     * @brief Q5_K tensor (5-bit K-quant super-block)
     */
    class Q5_KTensor : public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable
    {
    public:
        Q5_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~Q5_KTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q5_K; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Format conversion (TensorBase interface)
        void to_fp32(float *dst) const override { to_fp32_via_blocks(dst); }
        void to_bf16(uint16_t *dst) const override;
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override
        {
            return to_int8_perchannel_via_blocks(dst_int8, dst_col_scales, dst_row_scales);
        }
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;

        // Q8_0 quantization (for quantized GEMM)
        void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const;

        // View support (row-slice views for MPI partitioning)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        // ITensorGemmTileDataProvider interface (inlined for zero overhead)
        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q5_KBlock::BLOCK_SIZE - 1) / Q5_KBlock::BLOCK_SIZE;
            // Use view-aware data pointer
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q5_KBlock *blocks = reinterpret_cast<const Q5_KBlock *>(data_ptr);
            const Q5_KBlock &block = blocks[row_idx * blocks_per_row + k_block_offset];
            decodeBlock(block, output);
        }

        __attribute__((always_inline))
        const void *
        get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q5_KBlock::BLOCK_SIZE - 1) / Q5_KBlock::BLOCK_SIZE;
            // Use view-aware data pointer
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q5_KBlock *blocks = reinterpret_cast<const Q5_KBlock *>(data_ptr);
            return &blocks[row_idx * blocks_per_row + k_block_offset];
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return Q5_KBlock::BLOCK_SIZE; }

        // SIMD decode methods (public for testing)
        static void decodeBlock(const Q5_KBlock &block, float *output);
        static void decodeBlockScalar(const Q5_KBlock &block, float *output);
#if defined(__AVX512F__)
        static void decodeBlockAVX512(const Q5_KBlock &block, float *output);
#endif
#if defined(__AVX2__)
        static void decodeBlockAVX2(const Q5_KBlock &block, float *output);
#endif

    private:
        // View constructor (borrows parent's data)
        Q5_KTensor(
            const std::vector<size_t> &shape,
            const uint8_t *parent_raw_data,
            size_t byte_offset,
            std::shared_ptr<TensorBase> parent);

        std::vector<size_t> shape_;
        std::vector<uint8_t> raw_data_; // Owned only by parent tensor
        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;

        // View support fields
        bool is_view_;
        const uint8_t *raw_data_ptr_;        // Points to parent's raw_data_.data()
        size_t view_byte_offset_;            // Byte offset from raw_data_ptr_
        std::shared_ptr<TensorBase> parent_; // Keeps parent alive

        static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m);
    };

    // Implementation: Q3_KTensor.cpp
    /**
     * @brief Q3_K tensor (3-bit K-quant super-block)
     */
    class Q3_KTensor : public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable
    {
    public:
        Q3_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~Q3_KTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q3_K; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Format conversion (TensorBase interface)
        void to_fp32(float *dst) const override { to_fp32_via_blocks(dst); }
        void to_bf16(uint16_t *dst) const override;
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override
        {
            return to_int8_perchannel_via_blocks(dst_int8, dst_col_scales, dst_row_scales);
        }
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;

        // Q8_0 conversion API (for GEMM kernel compatibility)
        void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const;

        // View support (row-slice only, preserves 256-element super-block alignment)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q3_KBlock::BLOCK_SIZE - 1) / Q3_KBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q3_KBlock *blocks = reinterpret_cast<const Q3_KBlock *>(data_ptr);
            decodeBlock(blocks[row_idx * blocks_per_row + k_block_offset], output);
        }

        __attribute__((always_inline))
        const void *
        get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q3_KBlock::BLOCK_SIZE - 1) / Q3_KBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q3_KBlock *blocks = reinterpret_cast<const Q3_KBlock *>(data_ptr);
            return &blocks[row_idx * blocks_per_row + k_block_offset];
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return Q3_KBlock::BLOCK_SIZE; }

        // Public decode methods for testing
        static void decodeBlock(const Q3_KBlock &block, float *output);
        static void decodeBlockScalar(const Q3_KBlock &block, float *output);

#if defined(__AVX512F__)
        static void decodeBlockAVX512(const Q3_KBlock &block, float *output);
#endif

#if defined(__AVX2__)
        static void decodeBlockAVX2(const Q3_KBlock &block, float *output);
#endif

    private:
        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)

        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;

        // Private view constructor
        Q3_KTensor(const std::vector<size_t> &shape,
                   const uint8_t *parent_raw_data,
                   size_t byte_offset,
                   std::shared_ptr<TensorBase> parent);
    };

    // Implementation: Q4_KTensor.cpp
    /**
     * @brief Q4_K tensor (4-bit K-quant super-block)
     */
    class Q4_KTensor : public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable
    {
    public:
        Q4_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~Q4_KTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q4_K; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Format conversion (TensorBase interface)
        void to_fp32(float *dst) const override { to_fp32_via_blocks(dst); }
        void to_bf16(uint16_t *dst) const override;
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override
        {
            return to_int8_perchannel_via_blocks(dst_int8, dst_col_scales, dst_row_scales);
        }
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;

        // Q8_0 quantization (for quantized GEMM)
        void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const;

        // View support (row-slice views for MPI partitioning)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        // ITensorGemmTileDataProvider interface (inlined for zero overhead)
        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q4_KBlock::BLOCK_SIZE - 1) / Q4_KBlock::BLOCK_SIZE;
            // Use view-aware data pointer
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q4_KBlock *blocks = reinterpret_cast<const Q4_KBlock *>(data_ptr);
            const Q4_KBlock &block = blocks[row_idx * blocks_per_row + k_block_offset];
            decodeBlock(block, output);
        }

        __attribute__((always_inline))
        const void *
        get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q4_KBlock::BLOCK_SIZE - 1) / Q4_KBlock::BLOCK_SIZE;
            // Use view-aware data pointer
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q4_KBlock *blocks = reinterpret_cast<const Q4_KBlock *>(data_ptr);
            return &blocks[row_idx * blocks_per_row + k_block_offset];
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return Q4_KBlock::BLOCK_SIZE; }

        // SIMD decode methods (public for testing)
        static void decodeBlock(const Q4_KBlock &block, float *output);
        static void decodeBlockScalar(const Q4_KBlock &block, float *output);
#if defined(__AVX512F__)
        static void decodeBlockAVX512(const Q4_KBlock &block, float *output);
#endif
#if defined(__AVX2__)
        static void decodeBlockAVX2(const Q4_KBlock &block, float *output);
#endif

    private:
        // View constructor (borrows parent's data)
        Q4_KTensor(
            const std::vector<size_t> &shape,
            const uint8_t *parent_raw_data,
            size_t byte_offset,
            std::shared_ptr<TensorBase> parent);

        std::vector<size_t> shape_;
        std::vector<uint8_t> raw_data_; // Owned only by parent tensor
        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;

        // View support fields
        bool is_view_;
        const uint8_t *raw_data_ptr_;        // Points to parent's raw_data_.data()
        size_t view_byte_offset_;            // Byte offset from raw_data_ptr_
        std::shared_ptr<TensorBase> parent_; // Keeps parent alive

        static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m);
    };

    // Implementation: Q8_KTensor.cpp
    /**
     * @brief Q8_K tensor (8-bit K-quant super-block)
     */
    class Q8_KTensor : public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable
    {
    public:
        Q8_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~Q8_KTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q8_K; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Format conversion (TensorBase interface)
        void to_fp32(float *dst) const override { to_fp32_via_blocks(dst); }
        void to_bf16(uint16_t *dst) const override;
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override
        {
            return to_int8_perchannel_via_blocks(dst_int8, dst_col_scales, dst_row_scales);
        }
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;

        // IQ8_0Decodable interface - per-block decode to Q8_0 (used by Q8_0WeightAccessor)
        void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const override;

        // View support (row-slice views for MPI partitioning)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        // ITensorGemmTileDataProvider interface (inlined for zero overhead)
        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q8_KBlock::BLOCK_SIZE - 1) / Q8_KBlock::BLOCK_SIZE;
            // Use view-aware data pointer
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q8_KBlock *blocks = reinterpret_cast<const Q8_KBlock *>(data_ptr);
            const Q8_KBlock &block = blocks[row_idx * blocks_per_row + k_block_offset];
            decodeBlock(block, output);
        }

        __attribute__((always_inline))
        const void *
        get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q8_KBlock::BLOCK_SIZE - 1) / Q8_KBlock::BLOCK_SIZE;
            // Use view-aware data pointer
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q8_KBlock *blocks = reinterpret_cast<const Q8_KBlock *>(data_ptr);
            return &blocks[row_idx * blocks_per_row + k_block_offset];
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return Q8_KBlock::BLOCK_SIZE; }

        // SIMD decode methods (public for testing)
        static void decodeBlock(const Q8_KBlock &block, float *output);
        static void decodeBlockScalar(const Q8_KBlock &block, float *output);
#ifdef __AVX512F__
        static void decodeBlockAVX512(const Q8_KBlock &block, float *output);
#endif
#ifdef __AVX2__
        static void decodeBlockAVX2(const Q8_KBlock &block, float *output);
#endif

    private:
        // View constructor (borrows parent's data)
        Q8_KTensor(
            const std::vector<size_t> &shape,
            const uint8_t *parent_raw_data,
            size_t byte_offset,
            std::shared_ptr<TensorBase> parent);

        std::vector<size_t> shape_;
        std::vector<uint8_t> raw_data_; // Owned only by parent tensor
        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;

        // View support fields
        bool is_view_;
        const uint8_t *raw_data_ptr_;        // Points to parent's raw_data_.data()
        size_t view_byte_offset_;            // Byte offset from raw_data_ptr_
        std::shared_ptr<TensorBase> parent_; // Keeps parent alive
    };

    // ===== IQ Tensors =====

    // Implementation: IQ4_XSTensor.cpp
    /**
     * @brief IQ4_XS tensor (4-bit extra-small IQ)
     */
    class IQ4_XSTensor : public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable
    {
    public:
        IQ4_XSTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~IQ4_XSTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ4_XS; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Format conversion (TensorBase interface)
        void to_fp32(float *dst) const override { to_fp32_via_blocks(dst); }
        void to_bf16(uint16_t *dst) const override;
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override
        {
            return to_int8_perchannel_via_blocks(dst_int8, dst_col_scales, dst_row_scales);
        }
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;
        // Q8_0 conversion API (for GEMM kernel compatibility)
        void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const;

        // View support (row-slice only - preserves K dimension)
        bool is_view() const override { return is_view_; }
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        // ITensorGemmTileDataProvider interface (inline for zero overhead in GEMM hot path)
        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            const size_t blocks_per_row = (shape_[1] + IQ4_XSBlock::BLOCK_SIZE - 1) / IQ4_XSBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ4_XSBlock *blocks = reinterpret_cast<const IQ4_XSBlock *>(data_ptr);
            const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
            decodeBlock(blocks[block_idx], output);
        }

        __attribute__((always_inline))
        const void *
        get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t blocks_per_row = (shape_[1] + IQ4_XSBlock::BLOCK_SIZE - 1) / IQ4_XSBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ4_XSBlock *blocks = reinterpret_cast<const IQ4_XSBlock *>(data_ptr);
            const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
            return &blocks[block_idx];
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return IQ4_XSBlock::BLOCK_SIZE; }

        // SIMD decode methods (public for testing)
        static void decodeBlock(const IQ4_XSBlock &block, float *output);
        static void decodeBlockScalar(const IQ4_XSBlock &block, float *output);
#ifdef __AVX512F__
        static void decodeBlockAVX512(const IQ4_XSBlock &block, float *output);
#endif
#ifdef __AVX2__
        static void decodeBlockAVX2(const IQ4_XSBlock &block, float *output);
#endif

    private:
        IQ4_XSTensor(const std::vector<size_t> &shape, const uint8_t *raw_data_ptr,
                     size_t view_byte_offset, std::shared_ptr<TensorBase> parent);

        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)

        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
    };

    // Implementation: IQ2_XXSTensor.cpp
    /**
     * @brief IQ2_XXS tensor (2-bit extra-extra-small IQ)
     */
    class IQ2_XXSTensor : public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable
    {
    public:
        IQ2_XXSTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~IQ2_XXSTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ2_XXS; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Format conversion (TensorBase interface)
        void to_fp32(float *dst) const override { to_fp32_via_blocks(dst); }
        void to_bf16(uint16_t *dst) const override;
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override
        {
            return to_int8_perchannel_via_blocks(dst_int8, dst_col_scales, dst_row_scales);
        }
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;
        // Q8_0 conversion API (for GEMM kernel compatibility)
        void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const;

        // View support (row-slice only - preserves K dimension)
        bool is_view() const override { return is_view_; }
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        // ITensorGemmTileDataProvider interface (inline for zero overhead in GEMM hot path)
        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            const size_t blocks_per_row = (shape_[1] + IQ2_XXSBlock::BLOCK_SIZE - 1) / IQ2_XXSBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ2_XXSBlock *blocks = reinterpret_cast<const IQ2_XXSBlock *>(data_ptr);
            const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
            decodeBlock(blocks[block_idx], output);
        }

        __attribute__((always_inline))
        const void *
        get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t blocks_per_row = (shape_[1] + IQ2_XXSBlock::BLOCK_SIZE - 1) / IQ2_XXSBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ2_XXSBlock *blocks = reinterpret_cast<const IQ2_XXSBlock *>(data_ptr);
            const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
            return &blocks[block_idx];
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return IQ2_XXSBlock::BLOCK_SIZE; }

        // SIMD decode methods (public for testing)
        static void decodeBlock(const IQ2_XXSBlock &block, float *output);
        static void decodeBlockScalar(const IQ2_XXSBlock &block, float *output);
#ifdef __AVX512F__
        static void decodeBlockAVX512(const IQ2_XXSBlock &block, float *output);
#endif
#ifdef __AVX2__
        static void decodeBlockAVX2(const IQ2_XXSBlock &block, float *output);
#endif

    private:
        IQ2_XXSTensor(const std::vector<size_t> &shape, const uint8_t *raw_data_ptr,
                      size_t view_byte_offset, std::shared_ptr<TensorBase> parent);

        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)

        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
    };

    // Implementation: IQ2_XSTensor.cpp
    /**
     * @brief IQ2_XS tensor (2-bit extra-small IQ)
     */
    class IQ2_XSTensor : public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable
    {
    public:
        IQ2_XSTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~IQ2_XSTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ2_XS; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Format conversion (TensorBase interface)
        void to_fp32(float *dst) const override { to_fp32_via_blocks(dst); }
        void to_bf16(uint16_t *dst) const override;
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override
        {
            return to_int8_perchannel_via_blocks(dst_int8, dst_col_scales, dst_row_scales);
        }
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;
        // Q8_0 conversion API (for GEMM kernel compatibility)
        void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const;

        // View support (row-slice only - preserves K dimension)
        bool is_view() const override { return is_view_; }
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        // ITensorGemmTileDataProvider interface (inline for zero overhead in GEMM hot path)
        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            const size_t blocks_per_row = (shape_[1] + IQ2_XSBlock::BLOCK_SIZE - 1) / IQ2_XSBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ2_XSBlock *blocks = reinterpret_cast<const IQ2_XSBlock *>(data_ptr);
            const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
            decodeBlock(blocks[block_idx], output);
        }

        __attribute__((always_inline))
        const void *
        get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t blocks_per_row = (shape_[1] + IQ2_XSBlock::BLOCK_SIZE - 1) / IQ2_XSBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ2_XSBlock *blocks = reinterpret_cast<const IQ2_XSBlock *>(data_ptr);
            const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
            return &blocks[block_idx];
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return IQ2_XSBlock::BLOCK_SIZE; }

        // SIMD decode methods (public for testing)
        static void decodeBlock(const IQ2_XSBlock &block, float *output);
        static void decodeBlockScalar(const IQ2_XSBlock &block, float *output);
#ifdef __AVX512F__
        static void decodeBlockAVX512(const IQ2_XSBlock &block, float *output);
#endif
#ifdef __AVX2__
        static void decodeBlockAVX2(const IQ2_XSBlock &block, float *output);
#endif

    private:
        IQ2_XSTensor(const std::vector<size_t> &shape, const uint8_t *raw_data_ptr,
                     size_t view_byte_offset, std::shared_ptr<TensorBase> parent);

        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)

        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
    };

    // Implementation: IQ3_XXSTensor.cpp
    /**
     * @brief IQ3_XXS tensor (3-bit extra-extra-small IQ)
     */
    class IQ3_XXSTensor : public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable
    {
    public:
        IQ3_XXSTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~IQ3_XXSTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ3_XXS; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Format conversion (TensorBase interface)
        void to_fp32(float *dst) const override { to_fp32_via_blocks(dst); }
        void to_bf16(uint16_t *dst) const override;
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override
        {
            return to_int8_perchannel_via_blocks(dst_int8, dst_col_scales, dst_row_scales);
        }
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;
        // Q8_0 conversion API (for GEMM kernel compatibility)
        void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const;

        // View support (row-slice only - preserves K dimension)
        bool is_view() const override { return is_view_; }
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        // ITensorGemmTileDataProvider interface (inline for zero overhead in GEMM hot path)
        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            const size_t blocks_per_row = (shape_[1] + IQ3_XXSBlock::BLOCK_SIZE - 1) / IQ3_XXSBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ3_XXSBlock *blocks = reinterpret_cast<const IQ3_XXSBlock *>(data_ptr);
            const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
            decodeBlock(blocks[block_idx], output);
        }

        __attribute__((always_inline))
        const void *
        get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t blocks_per_row = (shape_[1] + IQ3_XXSBlock::BLOCK_SIZE - 1) / IQ3_XXSBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ3_XXSBlock *blocks = reinterpret_cast<const IQ3_XXSBlock *>(data_ptr);
            const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
            return &blocks[block_idx];
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return IQ3_XXSBlock::BLOCK_SIZE; }

        // SIMD decode methods (public for testing)
        static void decodeBlock(const IQ3_XXSBlock &block, float *output);
        static void decodeBlockScalar(const IQ3_XXSBlock &block, float *output);
#ifdef __AVX512F__
        static void decodeBlockAVX512(const IQ3_XXSBlock &block, float *output);
#endif
#ifdef __AVX2__
        static void decodeBlockAVX2(const IQ3_XXSBlock &block, float *output);
#endif

    private:
        IQ3_XXSTensor(const std::vector<size_t> &shape, const uint8_t *raw_data_ptr,
                      size_t view_byte_offset, std::shared_ptr<TensorBase> parent);

        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)

        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
    };

    // Implementation: IQ2_STensor.cpp
    /**
     * @brief IQ2_S tensor (2-bit small IQ)
     */
    class IQ2_STensor : public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable
    {
    public:
        IQ2_STensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~IQ2_STensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ2_S; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Format conversion (TensorBase interface)
        void to_fp32(float *dst) const override { to_fp32_via_blocks(dst); }
        void to_bf16(uint16_t *dst) const override;
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override
        {
            return to_int8_perchannel_via_blocks(dst_int8, dst_col_scales, dst_row_scales);
        }
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;
        // Q8_0 conversion API (for GEMM kernel compatibility)
        void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const;

        // View support (row-slice only - preserves K dimension)
        bool is_view() const override { return is_view_; }
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        // ITensorGemmTileDataProvider interface (inline for zero overhead in GEMM hot path)
        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            const size_t blocks_per_row = (shape_[1] + IQ2_SBlock::BLOCK_SIZE - 1) / IQ2_SBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ2_SBlock *blocks = reinterpret_cast<const IQ2_SBlock *>(data_ptr);
            const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
            decodeBlock(blocks[block_idx], output);
        }

        __attribute__((always_inline))
        const void *
        get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t blocks_per_row = (shape_[1] + IQ2_SBlock::BLOCK_SIZE - 1) / IQ2_SBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ2_SBlock *blocks = reinterpret_cast<const IQ2_SBlock *>(data_ptr);
            const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
            return &blocks[block_idx];
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return IQ2_SBlock::BLOCK_SIZE; }

        // SIMD decode methods (public for testing)
        static void decodeBlock(const IQ2_SBlock &block, float *output);
        static void decodeBlockScalar(const IQ2_SBlock &block, float *output);
#ifdef __AVX512F__
        static void decodeBlockAVX512(const IQ2_SBlock &block, float *output);
#endif
#ifdef __AVX2__
        static void decodeBlockAVX2(const IQ2_SBlock &block, float *output);
#endif

    private:
        IQ2_STensor(const std::vector<size_t> &shape, const uint8_t *raw_data_ptr,
                    size_t view_byte_offset, std::shared_ptr<TensorBase> parent);

        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)

        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
    };

    // Implementation: IQ3_STensor.cpp
    /**
     * @brief IQ3_S tensor (3-bit small IQ)
     */
    class IQ3_STensor : public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable
    {
    public:
        IQ3_STensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~IQ3_STensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ3_S; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Format conversion (TensorBase interface)
        void to_fp32(float *dst) const override { to_fp32_via_blocks(dst); }
        void to_bf16(uint16_t *dst) const override;
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override
        {
            return to_int8_perchannel_via_blocks(dst_int8, dst_col_scales, dst_row_scales);
        }
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;
        // Q8_0 conversion API (for GEMM kernel compatibility)
        void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const;

        // View support (row-slice only - preserves K dimension)
        bool is_view() const override { return is_view_; }
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        // ITensorGemmTileDataProvider interface (inline for zero overhead in GEMM hot path)
        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            const size_t blocks_per_row = (shape_[1] + IQ3_SBlock::BLOCK_SIZE - 1) / IQ3_SBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ3_SBlock *blocks = reinterpret_cast<const IQ3_SBlock *>(data_ptr);
            const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
            decodeBlock(blocks[block_idx], output);
        }

        __attribute__((always_inline))
        const void *
        get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t blocks_per_row = (shape_[1] + IQ3_SBlock::BLOCK_SIZE - 1) / IQ3_SBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ3_SBlock *blocks = reinterpret_cast<const IQ3_SBlock *>(data_ptr);
            const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
            return &blocks[block_idx];
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return IQ3_SBlock::BLOCK_SIZE; }

        // SIMD decode methods (public for testing)
        static void decodeBlock(const IQ3_SBlock &block, float *output);
        static void decodeBlockScalar(const IQ3_SBlock &block, float *output);
#ifdef __AVX512F__
        static void decodeBlockAVX512(const IQ3_SBlock &block, float *output);
#endif
#ifdef __AVX2__
        static void decodeBlockAVX2(const IQ3_SBlock &block, float *output);
#endif

    private:
        IQ3_STensor(const std::vector<size_t> &shape, const uint8_t *raw_data_ptr,
                    size_t view_byte_offset, std::shared_ptr<TensorBase> parent);

        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)

        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
    };

    // Implementation: IQ1_STensor.cpp
    /**
     * @brief IQ1_S tensor (1-bit small IQ)
     */
    class IQ1_STensor : public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable
    {
    public:
        IQ1_STensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~IQ1_STensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ1_S; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Format conversion (TensorBase interface)
        void to_fp32(float *dst) const override { to_fp32_via_blocks(dst); }
        void to_bf16(uint16_t *dst) const override;
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override
        {
            return to_int8_perchannel_via_blocks(dst_int8, dst_col_scales, dst_row_scales);
        }
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;
        // Q8_0 conversion API (for GEMM kernel compatibility)
        void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const;

        // View support (row-slice only - preserves K dimension)
        bool is_view() const override { return is_view_; }
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        // ITensorGemmTileDataProvider interface
        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            const size_t blocks_per_row = (shape_[1] + IQ1_SBlock::BLOCK_SIZE - 1) / IQ1_SBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ1_SBlock *blocks = reinterpret_cast<const IQ1_SBlock *>(data_ptr);
            const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
            decodeBlock(blocks[block_idx], output);
        }

        __attribute__((always_inline))
        const void *
        get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t blocks_per_row = (shape_[1] + IQ1_SBlock::BLOCK_SIZE - 1) / IQ1_SBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ1_SBlock *blocks = reinterpret_cast<const IQ1_SBlock *>(data_ptr);
            const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
            return &blocks[block_idx];
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return IQ1_SBlock::BLOCK_SIZE; }

        // SIMD decode methods (public for testing)
        static void decodeBlock(const IQ1_SBlock &block, float *output);
        static void decodeBlockScalar(const IQ1_SBlock &block, float *output);
#ifdef __AVX512F__
        static void decodeBlockAVX512(const IQ1_SBlock &block, float *output);
#endif
#ifdef __AVX2__
        static void decodeBlockAVX2(const IQ1_SBlock &block, float *output);
#endif

    private:
        IQ1_STensor(const std::vector<size_t> &shape, const uint8_t *raw_data_ptr,
                    size_t view_byte_offset, std::shared_ptr<TensorBase> parent);

        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)

        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
    };

    // Implementation: IQ1_MTensor.cpp
    /**
     * @brief IQ1_M tensor (1-bit medium IQ)
     */
    class IQ1_MTensor : public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable
    {
    public:
        IQ1_MTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~IQ1_MTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ1_M; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Format conversion (TensorBase interface)
        void to_fp32(float *dst) const override { to_fp32_via_blocks(dst); }
        void to_bf16(uint16_t *dst) const override;
        void to_fp16(uint16_t *dst) const override;
        void to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size = 32) const override;
        bool to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales = nullptr) const override
        {
            return to_int8_perchannel_via_blocks(dst_int8, dst_col_scales, dst_row_scales);
        }
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;
        // Q8_0 conversion API (for GEMM kernel compatibility)
        void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const;

        // View support (row-slice only - preserves K dimension)
        bool is_view() const override { return is_view_; }
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        // ITensorGemmTileDataProvider interface (inline for zero overhead in GEMM hot path)
        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            const size_t blocks_per_row = (shape_[1] + IQ1_MBlock::BLOCK_SIZE - 1) / IQ1_MBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ1_MBlock *blocks = reinterpret_cast<const IQ1_MBlock *>(data_ptr);
            const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
            decodeBlock(blocks[block_idx], output);
        }

        __attribute__((always_inline))
        const void *
        get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t blocks_per_row = (shape_[1] + IQ1_MBlock::BLOCK_SIZE - 1) / IQ1_MBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ1_MBlock *blocks = reinterpret_cast<const IQ1_MBlock *>(data_ptr);
            const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
            return &blocks[block_idx];
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return IQ1_MBlock::BLOCK_SIZE; }

        // SIMD decode methods (public for testing)
        static void decodeBlock(const IQ1_MBlock &block, float *output);
        static void decodeBlockScalar(const IQ1_MBlock &block, float *output);
#ifdef __AVX512F__
        static void decodeBlockAVX512(const IQ1_MBlock &block, float *output);
#endif
#ifdef __AVX2__
        static void decodeBlockAVX2(const IQ1_MBlock &block, float *output);
#endif

    private:
        IQ1_MTensor(const std::vector<size_t> &shape, const uint8_t *raw_data_ptr,
                    size_t view_byte_offset, std::shared_ptr<TensorBase> parent);

        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)

        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
    };

} // namespace llaminar2
