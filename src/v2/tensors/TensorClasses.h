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
 * 5. **IINT8Decodable** - Direct conversion to INT8 per-column symmetric quantization
 *    - Implemented by: Quantized weight tensors (Q4_0, IQ4_NL, Q6_K, Q8_0, etc.)
 *    - Usage: Fusion kernels (FusedDualGEMM, FusedTripleGEMM), model loader optimization
 *    - Key method: decode_to_int8_percol() - direct Q4_0→INT8 without FP32 intermediate
 *    - Optimization: Eliminates one quantization step, reduces error accumulation
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

#include "ITensor.h"
#include "TypedTensorBase.h" // CRTP base for type-safe typed_data() access
#include "TensorKernels.h"
#include "FP16Utils.h"
#include "BlockStructures.h"      // Must be included BEFORE SIMDHelpers.h
#include "TensorLayout.h"         // Tensor memory layout contracts
#include "NativeVnniFormatInfo.h" // Format metadata returned by IINT8Unpackable::vnniFormatInfo()
#include "VnniPackContext.h"      // Packing context for IINT8Unpackable::packVnniBlock()
#include "SIMDHelpers.h"
#include "AlignedVector.h"
#include "../backends/DeviceId.h" // DeviceId for ensureOnDevice
#include <vector>
#include <memory>
#include <cstddef>
#include <cstdint>
#include <any>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <optional>

namespace llaminar2
{
    // Forward declaration for backend dependency injection
    class IBackend;

    // All block structures are now defined in BlockStructures.h
    // This eliminates circular dependencies between Tensors.h and SIMDHelpers.h

    // TensorType enum and tensorTypeName() are now defined in TensorType.h
    // (included transitively via ITensor.h)

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
        virtual std::unique_ptr<ITensorEmbedding> createEmbedding() = 0;

        /**
         * @brief Quantize activations to INT8 with per-row scales
         * @param rows Number of rows (M dimension)
         * @param cols Number of columns (K dimension)
         * @return Packed activation data and per-row scales
         */
        virtual ActivationPack to_int8_activation_pack(int rows, int cols) const = 0;

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
         * @brief Apply RMS normalization in-place (native precision)
         *
         * Each tensor type implements this using its native precision method:
         * - FP32Tensor: Creates FP32 RMSNorm kernel
         * - BF16Tensor: Creates BF16 RMSNorm kernel (apply_bf16)
         * - FP16Tensor: Creates FP16 RMSNorm kernel (apply_fp16)
         *
         * @param gamma Gamma (scale) weights [d_model]
         * @param seq_len Sequence length (number of rows)
         * @param d_model Model dimension (columns)
         * @param eps Epsilon for numerical stability (default 1e-6)
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

        /**
         * @brief Bulk quantize activation tensor to Q8_1 format for GEMM
         *
         * Quantizes the entire activation buffer [m, k] to Q8_1 blocks for use
         * in integer GEMM operations. This is the optimized bulk quantization
         * path used by QuantisedGemmKernel.
         *
         * **Design Pattern ("quantize once, use many times")**:
         * - Activation quantized to Q8_1 once before multi-head attention
         * - Same Q8_1 blocks reused for Q, K, V projections
         * - Eliminates redundant FP32→Q8_1 conversion overhead
         *
         * **Implementation by tensor type**:
         * - FP32Tensor: Direct quantization (optimized AVX512/AVX2 path)
         * - BF16Tensor: Dequant to FP32, then quantize to Q8_1
         * - FP16Tensor: Dequant to FP32, then quantize to Q8_1
         * - Q8_1Tensor: Zero-copy (already in Q8_1 format)
         *
         * @param q8_1_buffer Output buffer for Q8_1 blocks [m * k_blocks] where k_blocks = (k+31)/32
         * @param m Number of rows (batch_size * seq_len)
         * @param k Number of columns (must match tensor's K dimension)
         * @return true on success, false on failure
         *
         * @note Caller must pre-allocate buffer of size: m * ((k+31)/32) * sizeof(Q8_1Block)
         * @note Uses OpenMP parallelization for large M, collapse(2) for small M
         */
        virtual bool quantize_to_q8_1(void *q8_1_buffer, int m, int k) const = 0;

        /**
         * @brief Get buffer size needed for quantize_to_q8_1 output
         *
         * @param m Number of rows
         * @param k Number of columns
         * @return Size in bytes needed for Q8_1 buffer
         */
        static size_t get_q8_1_buffer_size(int m, int k)
        {
            int k_blocks = (k + 31) / 32;
            return static_cast<size_t>(m) * k_blocks * sizeof(Q8_1Block);
        }
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
     * @brief INT8 unpackable interface for native quantized format → plain INT8 unpacking
     *
     * Used by blockwise INT8 GEMM kernels (e.g., OneDNN BRGEMM) that need quantized weights
     * unpacked to plain int8 values in their NATIVE quantization range (NO requantization).
     *
     * **Architecture Philosophy (Native Unpacking, NOT Requantization)**:
     * - Q4_0: Unpack 4-bit nibbles to int8 range [-8, 7] via `(nibble - 8)`
     * - Q6_K: Unpack 6-bit values to int8 range [-32, 31]
     * - IQ4_NL: Unpack to native IQ4_NL int8 range (non-linear quantization)
     * - Original block scales preserved and applied AFTER BRGEMM dot product
     *
     * **Why NOT requantize?**
     * - BRGEMM computes: `C = A × B` (int8 dot product → int32 accumulator)
     * - Scales applied after: `C_fp32 = C_int32 * scale_A * scale_B`
     * - Requantizing B loses precision: Q4_0 → FP32 → Q8_0 (double quantization)
     * - Native unpacking preserves original quantization: Q4_0 → int8 (single quantization)
     *
     * **BRGEMM Integration Pattern**:
     * ```cpp
     * // 1. Unpack weight block to plain int8 (native range)
     * int8_t B_unpacked[32];
     * weight->unpack_block_to_int8(row, kb, B_unpacked);
     *
     * // 2. BRGEMM matmul (u8 × s8 → s32)
     * brgemm.execute(A_u8, B_unpacked, C_s32);
     *
     * // 3. Apply original scales after matmul
     * float scale_A = activation->get_block_scale(...);
     * float scale_B = weight->get_block_scale(row, kb);
     * C_fp32[i] = C_s32[i] * scale_A * scale_B;
     * ```
     *
     * @note This is **unpacking**, not requantization. Output is in NATIVE format range.
     */
    class IINT8Unpackable
    {
    public:
        virtual ~IINT8Unpackable() = default;

        /**
         * @brief Get native-VNNI format metadata for this tensor type
         *
         * Returns intrinsic encoding properties (codebook ID, payload size,
         * symmetry, etc.) that describe how this format is packed for GPU VNNI
         * kernels. Only native-VNNI formats (≤6-bit) return non-null.
         *
         * @return Pointer to static metadata, or nullptr for non-VNNI formats (Q8_0, Q8_1)
         */
        virtual const NativeVnniFormatInfo *vnniFormatInfo() const { return nullptr; }

        /**
         * @brief Pack one 32-element block into interleaved VNNI layout
         *
         * Extracts payload bytes, FP16 scale (and min for asymmetric formats)
         * from one logical 32-element block and writes them to the interleaved
         * output arrays in the VnniPackContext.  Super-block formats decompose
         * on the fly using (b / 8, b % 8) addressing.
         *
         * Default is a no-op for non-VNNI formats (Q8_0, Q8_1).
         *
         * @param ctx  Packing context with output buffers and layout parameters
         * @param n    Row index (output feature)
         * @param b    Block index within the row (0 to blocks_per_row-1)
         */
        virtual void packVnniBlock(const VnniPackContext &ctx, int n, int b) const {}

        /**
         * @brief Unpack one quantized block to plain int8 values (native range, NO requantization)
         *
         * Unpacks native format (Q4_0, Q6_K, IQ4_NL) to plain int8 values in their
         * native quantization range. Does NOT compute new scales or requantize.
         *
         * @param row_idx Row index in weight tensor
         * @param k_block_offset Block offset along K dimension (units of block_size())
         * @param output Output buffer for plain int8 values (must have space for block_size() elements)
         *
         * Examples:
         * - Q4_0: Unpacks 4-bit nibbles to range [-8, 7] via `(nibble - 8)`
         * - Q6_K: Unpacks 6-bit values to range [-32, 31]
         * - IQ4_NL: Unpacks to native IQ4_NL int8 range
         *
         * @note Output values are in NATIVE range (e.g., [-8,7] for Q4_0), NOT [-127,127]
         * @note Use get_block_scale() to retrieve original quantization scale
         */
        virtual void unpack_block_to_int8(
            size_t row_idx,
            size_t k_block_offset,
            int8_t *output) const = 0;

        /**
         * @brief Get original quantization scale for block (from native format)
         *
         * Returns the original scale factor used when the block was quantized.
         * This scale should be applied AFTER BRGEMM matmul, not before unpacking.
         *
         * @param row_idx Row index in weight tensor
         * @param k_block_offset Block offset along K dimension
         * @return FP32 scale factor from original quantization (e.g., Q4_0 block.d)
         *
         * @note For formats with per-block scales (Q4_0, Q8_0), returns block.d (FP16→FP32)
         * @note For formats with super-block scales (Q6_K), returns appropriate scale for this block
         */
        virtual float get_block_scale(
            size_t row_idx,
            size_t k_block_offset) const = 0;

        /**
         * @brief Get original quantization min (offset) for block
         *
         * Returns the minimum value (offset) used when the block was quantized.
         * Used for asymmetric quantization (e.g., Q4_1, Q5_1, Q_K).
         * Formula: value = scale * quant + min
         *
         * @param row_idx Row index in weight tensor
         * @param k_block_offset Block offset along K dimension
         * @return FP32 min value (default 0.0f for symmetric formats)
         */
        virtual float get_block_min(
            size_t row_idx,
            size_t k_block_offset) const { return 0.0f; }

        /**
         * @brief Get the super-block size for this format
         *
         * Returns 32 for simple formats (Q4_0, Q8_0, etc.) where each block is independent.
         * Returns 256 for K-quant and IQuant formats that have 256-element super-blocks
         * containing 8 sub-blocks of 32 elements each.
         *
         * @return 32 for simple formats, 256 for super-block formats
         */
        virtual size_t superblock_size() const { return 32; }

        /**
         * @brief Unpack an entire super-block to plain int8 values
         *
         * For super-block formats (Q4_K, Q6_K, IQ3_S, etc.), this unpacks all 256 elements
         * of a super-block in one call, which is more efficient than 8 separate calls to
         * unpack_block_to_int8() because:
         * - Super-block header (scales, mins) is read once instead of 8 times
         * - Better instruction-level parallelism
         * - Fewer function call overhead
         *
         * For simple 32-element formats, the default implementation just calls
         * unpack_block_to_int8() once.
         *
         * @param row_idx Row index in weight tensor
         * @param superblock_idx Super-block index along K dimension
         * @param output Output buffer (must have space for superblock_size() elements)
         * @param scales Output buffer for 8 sub-block scales (must have space for 8 floats, or nullptr)
         * @param mins Output buffer for 8 sub-block mins (must have space for 8 floats, or nullptr)
         *
         * @note For simple formats, superblock_idx == k_block_offset
         * @note For super-block formats, superblock_idx indexes 256-element super-blocks
         */
        virtual void unpack_superblock_to_int8(
            size_t row_idx,
            size_t superblock_idx,
            int8_t *output,
            float *scales = nullptr,
            float *mins = nullptr) const
        {
            // Default implementation for simple 32-element formats
            // Just calls unpack_block_to_int8 once
            unpack_block_to_int8(row_idx, superblock_idx, output);
            if (scales)
            {
                scales[0] = get_block_scale(row_idx, superblock_idx);
            }
            if (mins)
            {
                mins[0] = get_block_min(row_idx, superblock_idx);
            }
        }

        /**
         * @brief Requantize one row from native format to per-row INT8 [-127, 127]
         *
         * Computes a per-row scale and requantizes all blocks without materializing
         * a full FP32 intermediate buffer.  Each element becomes:
         *     output[i] = clamp(round(dequant_value[i] / row_scale), -127, 127)
         * where row_scale = max_abs / 127.
         *
         * Override for format-specific optimizations; the default implementation
         * uses unpack_block_to_int8() + get_block_scale() + get_block_min().
         *
         * @param row_idx Row index in the tensor
         * @param K       Number of columns (must be divisible by 32)
         * @param output  Output buffer (K elements)
         * @return Per-row scale factor (max_abs / 127)
         */
        virtual float requantizeRowToInt8(
            size_t row_idx,
            size_t K,
            int8_t *output) const
        {
            const size_t blocks_per_row = K / 32;

            // Phase 1: Compute max absolute dequantized value across all blocks
            float max_abs = 0.0f;
            alignas(64) int8_t tmp[32];
            for (size_t b = 0; b < blocks_per_row; ++b)
            {
                unpack_block_to_int8(row_idx, b, tmp);
                const float scale = std::abs(get_block_scale(row_idx, b));
                const float min_val = get_block_min(row_idx, b);
                for (int i = 0; i < 32; ++i)
                {
                    float val = std::abs(static_cast<float>(tmp[i]) * scale + min_val);
                    if (val > max_abs)
                        max_abs = val;
                }
            }

            const float row_scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
            const float inv_row_scale = 1.0f / row_scale;

            // Phase 2: Requantize each block to INT8
            for (size_t b = 0; b < blocks_per_row; ++b)
            {
                unpack_block_to_int8(row_idx, b, tmp);
                const float scale = get_block_scale(row_idx, b);
                const float min_val = get_block_min(row_idx, b);
                const float rescale = scale * inv_row_scale;
                const float min_offset = min_val * inv_row_scale;

                int8_t *dst = output + b * 32;
                for (int i = 0; i < 32; ++i)
                {
                    float val = static_cast<float>(tmp[i]) * rescale + min_offset;
                    int32_t q = static_cast<int32_t>(val + (val >= 0.0f ? 0.5f : -0.5f));
                    q = q < -127 ? -127 : (q > 127 ? 127 : q);
                    dst[i] = static_cast<int8_t>(q);
                }
            }

            return row_scale;
        }
    };

    /**
     * @brief Abstract tensor interface
     *
     * TensorBase extends ITensor to provide the common functionality needed by all tensor types:
     * - Device management (upload, download, affinity)
     * - Kernel factory methods (GEMM, RoPE, etc.)
     * - Format conversion (to_fp32, to_bf16, etc.)
     * - View support
     *
     * The ITensor interface provides:
     * - Runtime type introspection (native_type_id, is<T>, typed_as<T>)
     * - Raw data access (raw_data, raw_mutable_data)
     * - Basic shape/size queries
     *
     * **Diamond Inheritance Note**:
     * TensorBase uses virtual inheritance from ITensor to support diamond inheritance
     * when combined with TypedTensorBase<Derived, DataType>:
     *
     *                    ITensor (virtual)
     *                   /               \
     *     TypedTensorBase<T,D>        TensorBase
     *                   \               /
     *                    FP32Tensor (etc.)
     *
     * This allows concrete tensors to inherit type-safe typed_data() from TypedTensorBase
     * while still getting the full TensorBase infrastructure.
     */
    class TensorSlice; // Forward declaration for friend

    class TensorBase : public virtual ITensor, public std::enable_shared_from_this<TensorBase>
    {
        friend class TensorSlice; // Allow TensorSlice to access protected byte_size()/raw_host_data_ptr()

    public:
        virtual ~TensorBase(); // Implemented in TensorBase.cpp - clears KernelFactory cache

        // Non-copyable: tensors contain mutexes and cached kernels
        // Use copyFrom() method to copy data between tensors
        TensorBase(const TensorBase &) = delete;
        TensorBase &operator=(const TensorBase &) = delete;

        // Move is also disabled due to enable_shared_from_this semantics
        TensorBase(TensorBase &&) = delete;
        TensorBase &operator=(TensorBase &&) = delete;

        // Generic cache for CPU kernel state (e.g. packed VNNI weights)
        mutable std::any cache_;

        // Generic cache for CUDA kernel state (e.g. packed INT8 weights)
        // Separate from cache_ to allow both CPU and CUDA paths to coexist
        mutable std::any cuda_cache_;

        // Generic cache for ROCm kernel state (e.g. packed INT8 weights for CK)
        // Separate from cache_ and cuda_cache_ to allow CPU, CUDA, and ROCm paths to coexist
        mutable std::any rocm_cache_;

        // Synchronizes cache_ / cuda_cache_ / rocm_cache_ initialization and reset
        mutable std::mutex packed_cache_mutex_;

        // Shape and type - each concrete tensor class has its own shape_ member
        // and overrides shape() to return it.
        virtual TensorType native_type() const = 0;

        // ===== ITensor Interface Implementation =====

        /**
         * @brief Get runtime type ID for this tensor
         * @return Integer type ID matching TensorTypeId constants
         * @note Derived from native_type() enum value
         */
        int native_type_id() const override { return static_cast<int>(native_type()); }

        /**
         * @brief Get total number of logical elements
         * @return Product of all shape dimensions
         * @note Implements ITensor::numel() via existing element_count()
         */
        size_t numel() const override { return element_count(); }

        /**
         * @brief Get size in bytes of the raw data buffer
         * @return Number of bytes for underlying storage
         * @note Implements ITensor::size_bytes() - delegates to byte_size()
         * @note Override in derived class for quantized formats with different byte_size()
         */
        size_t size_bytes() const override { return byte_size(); }

        /**
         * @brief Get raw pointer to data buffer (const)
         * @return Void pointer to start of data buffer
         * @note Implements ITensor::raw_data() - delegates to raw_host_data_ptr()
         * @note Always returns HOST pointer even if tensor is on GPU
         */
        const void *raw_data() const override { return raw_host_data_ptr(); }

        /**
         * @brief Get raw mutable pointer to data buffer
         * @return Void pointer to start of data buffer
         * @note Implements ITensor::raw_mutable_data() - delegates to raw_host_data_ptr()
         * @note Always returns HOST pointer even if tensor is on GPU
         * @note IMPORTANT: Invalidates GPU copy since host will be modified
         */
        void *raw_mutable_data() override
        {
            invalidateGpuData(); // Host is about to be modified, GPU copy is now stale
            return raw_host_data_ptr();
        }

        /**
         * @brief Get pointer to data where it currently resides (const)
         * @return GPU pointer if isOnGPU(), else host pointer
         * @note Use this when dispatching to kernels that need the "active" pointer
         */
        const void *active_data_ptr() const override
        {
            return gpu_data_ptr_ ? gpu_data_ptr_ : raw_host_data_ptr();
        }

        /**
         * @brief Get mutable pointer to data where it currently resides
         * @return GPU pointer if isOnGPU(), else host pointer
         * @note Use this when dispatching to kernels that need the "active" pointer
         */
        void *active_mutable_data_ptr() override
        {
            return gpu_data_ptr_ ? gpu_data_ptr_ : raw_host_data_ptr();
        }

        // ===== Device Affinity API =====
        //
        // Device indices are DeviceManager indices:
        //   - NOT_ON_GPU (-1): Data is on host/CPU only, not on any GPU
        //   - >= 0: DeviceManager device index (use dm.devices()[idx] to get details)
        //
        // Note: DeviceManager index 0 is typically CPU. GPU indices start at 1+.
        // NOT_ON_GPU is distinct from "CPU device" - it means "no GPU copy exists".
        //
        // There are TWO device tracking concepts:
        //   1. "Home DM device" (home_dm_device_index) - where tensor was CREATED
        //   2. "Current DM device" (current_dm_device_index) - where GPU data currently RESIDES
        //
        // After ensureOnDevice(gpu_idx):
        //   - home_dm_device_index() still returns original value (e.g., NOT_ON_GPU for CPU-created)
        //   - current_dm_device_index() returns gpu_idx (where data now is)
        //   - is_on_device(gpu_idx) returns true
        //
        // Use current_dm_device_index() when checking WHERE GPU data is.
        // Use home_dm_device_index() when checking WHERE tensor was created/belongs.

        /** @brief Sentinel value meaning "not resident on any GPU" */
        static constexpr int NOT_ON_GPU = -1;

        /**
         * @brief Get the device where this tensor's data resides
         * @return DeviceId identifying CPU, CUDA, or ROCm device
         * @note Override in derived classes
         */
        virtual DeviceId home_device() const = 0;

        /**
         * @brief Get "home" DeviceManager device index where tensor was created
         *
         * This is the CREATION device, NOT the current location of data.
         * After ensureOnDevice(), this still returns the original value.
         *
         * @return NOT_ON_GPU (-1) for host/CPU, >=0 for DeviceManager device index
         * @note To check where data currently IS, use current_device()
         * @see current_device() for checking current GPU data location
         */

        /**
         * @brief Get the DeviceId where tensor GPU data currently resides
         *
         * This reflects the CURRENT GPU location of tensor data:
         * - After ensureOnDevice(device): returns device
         * - After ensureOnHost(): returns nullopt
         * - If never uploaded to GPU: returns nullopt
         *
         * For dual-residency (data on both CPU and GPU), returns GPU device.
         *
         * @return std::optional<DeviceId> - nullopt for host/CPU only, DeviceId for GPU
         * @note This is non-virtual - uses TensorBase's gpu_device_ tracking
         */
        std::optional<DeviceId> current_device() const { return gpu_device_; }

        // ===== Multi-Device Coherence API =====

        /**
         * @brief Get the device that currently has authoritative data
         * @return DeviceId if a GPU is authoritative, nullopt if host is authoritative
         */
        std::optional<DeviceId> getAuthoritativeDevice() const { return authoritative_device_; }

        /**
         * @brief Check if host memory is authoritative (has current data)
         *
         * Returns true when:
         * - Tensor was just created (default state)
         * - After ensureOnHost() synced from GPU
         * - Tensor has never been uploaded to GPU
         */
        bool isHostAuthoritative() const { return !authoritative_device_.has_value(); }

        /**
         * @brief Check if a specific device is authoritative
         * @param device Device to check
         * @return true if that device has the authoritative data
         */
        bool isDeviceAuthoritative(DeviceId device) const
        {
            return authoritative_device_.has_value() && *authoritative_device_ == device;
        }

        /**
         * @brief Transfer tensor data directly to another GPU device
         *
         * Uses ICollectiveBackend::copy() for direct P2P/BAR transfer.
         * Does NOT go through host staging - this is a direct GPU-to-GPU copy.
         *
         * @param dst_device Target GPU device
         * @return true on success, false if no backend supports this transfer
         *
         * @pre Tensor must have authoritative data on a GPU (call mark_device_dirty() first)
         * @post getAuthoritativeDevice() == dst_device
         * @post Source device buffer becomes stale
         * @post Host buffer becomes stale (if it exists)
         *
         * @note Fails fast if no backend supports the transfer path.
         *       Does NOT silently fall back to host staging.
         *
         * @example
         *   tensor->ensureOnDevice(cuda0);
         *   tensor->mark_device_dirty();  // GPU computed new values
         *   tensor->transferTo(rocm0);    // Direct transfer, no host staging
         */
        bool transferTo(DeviceId dst_device, size_t bytes_override = 0);

        /**
         * @brief Copy tensor data to another GPU, keeping both devices valid
         *
         * Unlike transferTo(), this keeps the source device buffer valid.
         * Useful for read-only sharing or when source needs to continue computing.
         *
         * @param dst_device Target GPU device
         * @return true on success
         *
         * @post Both src and dst devices have valid data
         * @post authoritative_device_ unchanged (source still authoritative)
         */
        bool copyTo(DeviceId dst_device);

        /**
         * @brief Check if tensor currently has valid data on the specified device
         *
         * @param device DeviceId to check
         * @return true if tensor has valid data on the specified device
         *
         * @note This checks CURRENT location, not creation device.
         *       A tensor can be on multiple devices simultaneously (dual-residency).
         */
        virtual bool is_on_device(DeviceId device) const
        {
            if (device.is_cpu())
            {
                // Asking about CPU/host - return true if host data is valid
                return host_valid_;
            }
            // Asking about a specific GPU - must be on that device AND have valid data
            return gpu_device_.has_value() && *gpu_device_ == device && device_valid_;
        }

        // ===== Lazy Transfer API (Phase 1 GPU Device-Aware Slicing) =====
        // These are NON-VIRTUAL - single implementation in TensorBase.cpp
        // Each tensor type just implements raw_host_data_ptr() and byte_size() accessors.

        /**
         * @brief Ensure tensor data is available on target device
         *
         * Performs lazy upload: if data is already on target device, returns immediately.
         * If data is on host, allocates GPU memory and uploads.
         *
         * @param target_device DeviceId for the target GPU device
         * @return true if data is now available on target device, false on error
         *
         * **Thread Safety**: Caller must ensure no concurrent modifications to tensor data
         * **Memory**: GPU buffer is allocated if not present, host buffer retained (dual-residency)
         *
         * @note For quantized weight tensors, uploads quantized blocks (not dequantized FP32)
         * @note Implementation in TensorBase.cpp - uses raw_host_data_ptr() and byte_size()
         */
        virtual bool ensureOnDevice(DeviceId target_device);

        /**
         * @brief Allocate GPU buffer without uploading host data
         *
         * Only allocates device memory - does NOT copy host data to GPU.
         * Use this for OUTPUT tensors where the kernel will overwrite the contents.
         *
         * @param target_device DeviceId for the target GPU device
         * @return true if allocation succeeded, false on error
         *
         * **Thread Safety**: Caller must ensure no concurrent modifications to tensor data
         * **Memory**: GPU buffer is allocated if not present; reuses existing if already on target
         * **Coherence**: Sets device_valid_=false (kernel will write), host_valid_ unchanged
         *
         * Compared to ensureOnDevice():
         *   - ensureOnDevice(): allocate + H2D upload (for INPUT tensors)
         *   - allocateOnDevice(): allocate only (for OUTPUT tensors)
         *
         * @note Implementation in TensorBase.cpp
         */
        virtual bool allocateOnDevice(DeviceId target_device);

        /**
         * @brief Invalidate GPU data, forcing re-upload on next ensureOnDevice() call
         *
         * Call this when host data has been modified and GPU copy is now stale.
         * This frees the GPU memory, so next ensureOnDevice() will reallocate and upload.
         */
        virtual void invalidateGpuData();

        /**
         * @brief Ensure tensor data is available on host (CPU)
         *
         * Performs lazy download: if data is already on host and valid, returns immediately.
         * If GPU has newer data (device_dirty_), downloads to host buffer.
         *
         * @return true if data is now available on host, false on error
         *
         * **Thread Safety**: Caller must ensure no concurrent modifications to tensor data
         * **Memory**: Host buffer is always retained; GPU buffer optionally freed (see releaseDeviceMemory)
         * @note Implementation in TensorBase.cpp - uses raw_host_data_ptr() and byte_size()
         */
        virtual bool ensureOnHost();

        /**
         * @brief Check if tensor data is currently resident on GPU
         * @return true if GPU buffer is allocated and contains valid data
         * @note Uses virtual gpu_data_ptr() so derived classes with custom device management work
         */
        bool isOnGPU() const { return gpu_data_ptr() != nullptr; }

        /**
         * @brief Get the raw device (GPU) data pointer
         * @return void* pointer to device memory, or nullptr if not on GPU
         * @note Use this for CUDA/ROCm kernel dispatch - the pointer is device-side memory
         * @note Virtual to allow tensor types with their own device management to override
         */
        virtual void *gpu_data_ptr();
        virtual const void *gpu_data_ptr() const;

        /**
         * @brief Mark tensor as modified on device (requires sync to host before host access)
         * @note Call this after GPU kernels write to the tensor via gpu_data_ptr()
         * @note This forwards to mark_device_dirty_with_event() for event-based sync
         *
         * After calling this:
         *   - device_valid_ = true (device just got written to)
         *   - host_valid_ = false (host is now stale) -- UNLESS tensor is mapped
         *   - A completion event is recorded for fine-grained synchronization
         *
         * For mapped tensors, both host and device stay valid since they share memory,
         * but we set mapped_needs_sync_ so ensureOnHost() knows to synchronize.
         */
        virtual void mark_device_dirty()
        {
            // Forward to event-based implementation for fine-grained sync
            mark_device_dirty_with_event();
        }

        /**
         * @brief Mark tensor as modified on device WITHOUT recording a completion event
         *
         * Lightweight version for intermediate tensors that only flow GPU→GPU on the
         * same compute stream. Skips hipEventRecord/cudaEventRecord overhead (~100-300μs
         * per call on ROCm). If data() is later called, ensureOnHost() will fall back to
         * a full device synchronize since no event is available.
         *
         * Use this for intermediate pipeline tensors. Use mark_device_dirty_with_event()
         * for tensors that will be read back to CPU (e.g., logits for sampling).
         *
         * @note Thread-safe (acquires coherence_mutex_)
         */
        void mark_device_dirty_flags_only()
        {
            std::lock_guard<std::mutex> lock(coherence_mutex_);
            device_valid_ = true;
            host_valid_ = is_mapped_;
            authoritative_device_ = gpu_device_;
            if (is_mapped_)
                mapped_needs_sync_ = true;
        }

        /**
         * @brief Mark tensor as modified on device and record a completion event
         *
         * This is the preferred method when fine-grained synchronization is desired.
         * Records an event after kernel execution so ensureOnHost() can wait on
         * just this kernel rather than doing a full device synchronization.
         *
         * @param stream Opaque GPU stream handle where the kernel ran.
         *               Pass the compute stream so the event is recorded on the
         *               correct stream. nullptr = default stream (stream 0).
         * @note Call this after GPU kernels write to the tensor via gpu_data_ptr()
         * @note If event recording fails, falls back to eventless behavior
         */
        virtual void mark_device_dirty_with_event(void *stream = nullptr);

        /**
         * @brief Clear the device completion event without destroying it through the backend
         *
         * This is used during cross-device transfers (e.g., CUDA -> ROCm) where the
         * event was created by a different backend. We can't destroy a CUDA event
         * through the ROCm backend, so we just clear the pointer. The event will
         * leak, but this only happens during PP transfers which are rare.
         *
         * @note Use this when transferring tensors between different GPU types
         *       to avoid passing CUDA events to ROCm's hipEventRecord.
         */
        void clearCompletionEvent()
        {
            device_completion_event_ = nullptr;
            event_device_.reset();
        }

        /**
         * @brief Check if tensor data is currently resident on host (CPU)
         * @return true if host buffer contains valid data
         */
        bool isOnCPU() const { return host_valid_; }

        /**
         * @brief Check if tensor data is currently valid on GPU
         * @return true if GPU buffer is allocated AND contains valid data
         */
        bool isDeviceValid() const { return device_valid_ && gpu_data_ptr_ != nullptr; }

        /**
         * @brief Check if tensor has cached device data via internal packing mechanisms
         *
         * CUDA/ROCm GEMM kernels often convert quantized weights to a different
         * representation (e.g., Q8_0 -> INT8 + per-column scales) and upload that
         * instead of the raw tensor data. This method checks if such cached data exists.
         *
         * Use this to skip redundant ensureOnDevice() calls for weight tensors
         * that use internal packing - they don't need raw tensor uploads.
         *
         * @param device_type Device type to check (CUDA or ROCm)
         * @return true if tensor has cached device data uploaded for the given device type
         * @note Implementation checks cuda_cache_/rocm_cache_ for uploaded packed weights
         */
        virtual bool hasCachedDeviceData(DeviceType device_type) const;

        /**
         * @brief Check if tensor uses zero-copy mapped memory
         * @return true if tensor was allocated with mapped memory (shared host/device)
         *
         * When a tensor is mapped:
         * - ensureOnDevice() and ensureOnHost() are no-ops
         * - GPU reads/writes directly to host memory via PCIe
         * - Both host_valid_ and device_valid_ are always true
         * - Eliminates sync memcpy bottleneck
         */
        bool isMapped() const { return is_mapped_; }

        /**
         * @brief Notify a mapped tensor that the GPU stream has been externally
         *        synchronized, so no per-access event wait is needed.
         *
         * Call this after performing a stream-level synchronization (e.g.,
         * hipStreamSynchronize) to avoid redundant hipEventSynchronize calls
         * when the host subsequently reads the mapped pointer via data().
         *
         * This is the preferred pattern for forward pass boundaries: the
         * orchestrator syncs the stream once, then marks logits as synced,
         * so the sampler receives logits without any coherence overhead.
         *
         * @note Only meaningful for mapped tensors (is_mapped_ == true).
         *       No-op for non-mapped tensors.
         */
        void markMappedSynced()
        {
            if (is_mapped_)
            {
                mapped_needs_sync_ = false;
            }
        }

        /**
         * @brief Check if tensor uses BAR-backed memory (PCIeBAR cross-vendor zero-copy)
         * @return true if tensor data resides in AMD VRAM accessed via PCIe BAR
         *
         * When a tensor is BAR-backed:
         * - Buffer physically resides in AMD VRAM (not system DRAM)
         * - CUDA writes go directly to AMD VRAM via PCIe (~2.65 GB/s)
         * - AMD accesses locally (full VRAM bandwidth)
         * - Both CUDA and ROCm see the same physical memory
         * - Host access via mmap (slow, uncached)
         *
         * See BARBackedTensor.h for detailed design documentation.
         */
        bool isBARBacked() const { return is_bar_backed_; }

        /**
         * @brief Get the ROCm-accessible pointer to tensor data for HIP kernels
         *
         * For BAR-backed tensors, this returns the HIP staging buffer pointer
         * (allocated via hipMalloc) that ROCm kernels can safely write to.
         *
         * CRITICAL: This does NOT return the BAR mmap address! HIP kernels cannot
         * directly dereference BAR mmap addresses (causes memory access fault).
         * The staging buffer is later copied to BAR via hipMemcpy(D2D).
         *
         * For non-BAR-backed tensors, returns nullptr.
         *
         * @return void* HIP device pointer for kernel writes, or nullptr if not BAR-backed
         */
        void *rocm_data_ptr() { return hip_staging_ptr_; }
        const void *rocm_data_ptr() const { return hip_staging_ptr_; }

        /**
         * @brief Get the raw BAR mmap pointer (for hipMemcpy D2D only)
         *
         * Returns the BAR mmap address for use with hipMemcpy(D2D).
         * DO NOT pass this to HIP kernels - they cannot dereference it!
         *
         * @return void* BAR mmap address, or nullptr if not BAR-backed
         */
        void *bar_address() { return bar_rocm_ptr_; }
        const void *bar_address() const { return bar_rocm_ptr_; }

        /**
         * @brief Get the CUDA device pointer for BAR memory
         *
         * Returns the CUDA-accessible pointer obtained via cuMemHostGetDevicePointer().
         * This pointer MUST be used for CUDA kernel access to BAR memory, not bar_address().
         *
         * @return void* CUDA device pointer to BAR, or nullptr if not BAR-backed
         */
        void *bar_cuda_ptr() { return bar_cuda_device_ptr_; }
        const void *bar_cuda_ptr() const { return bar_cuda_device_ptr_; }

        /**
         * @brief Initialize BAR-backed state with direct pointers
         *
         * Sets up BAR-backed memory state using pre-obtained pointers from DirectP2PEngine.
         * Unlike initBARBackedMemory(), this does NOT allocate - it only configures state.
         * The caller is responsible for ensuring the pointers are valid and properly aligned.
         *
         * This is a PUBLIC method intended for use by TensorFactory::createFP32BARBacked()
         * when working directly with DirectP2PEngine (without PCIeBARBackend).
         *
         * @param rocm_ptr ROCm-accessible pointer (mmap'd BAR = AMD VRAM)
         * @param cuda_ptr CUDA-accessible pointer (from cuMemHostGetDevicePointer)
         * @param rocm_device ROCm device whose BAR hosts this buffer
         * @param cuda_device CUDA device that has registered access
         * @param bytes Size of the allocation in bytes
         *
         * @note Does NOT take ownership of memory - no deallocation on destruction
         * @note Used by TensorFactory::createFP32BARBacked() with DirectP2PEngine
         */
        void initBARBackedDirect(void *rocm_ptr, void *cuda_ptr,
                                 DeviceId rocm_device, DeviceId cuda_device,
                                 size_t bytes);

        /**
         * @brief Release GPU memory, keeping only host copy
         *
         * Downloads data to host if needed, then frees GPU buffer.
         * After this call, isOnGPU() returns false.
         *
         * @return true on success, false on error
         * @note Implementation in TensorBase.cpp
         */
        bool releaseDeviceMemory();

        // Data access (fallback for non-fused operations)
        // NOTE: Returns pointer in native type - use convertTo<T>() for type conversion
        virtual const float *data() const = 0; // Returns host pointer (syncs from device if needed)
        virtual float *mutable_data() = 0;     // Returns host pointer, marks dirty

        // =========================================================================
        // Unified FP32 Data Access Interface (Phase 1 Infrastructure)
        // =========================================================================

        /**
         * @brief Explicit FP32 dequantization accessor
         *
         * For most tensor types, this is equivalent to data(). For Q8_1Tensor,
         * this performs explicit dequantization while data() throws to prevent
         * accidental implicit dequantization that defeats the purpose of Q8_1.
         *
         * Use this method when you INTENTIONALLY need FP32 data from a Q8_1 tensor
         * (e.g., for snapshot capture, debugging, or interop with FP32-only code).
         *
         * @return Const pointer to FP32 data (may be cached/lazily computed)
         * @note Default implementation calls data() - overridden by Q8_1Tensor
         */
        virtual const float *fp32_data() const { return data(); }

        /**
         * @brief Returns mutable pointer to FP32 data for activation tensors.
         *
         * For FP32Tensor, TensorSlice (of FP32), FP16Tensor, BF16Tensor:
         * returns mutable pointer.
         * For quantized weight tensors: returns nullptr (weights are read-only).
         *
         * @return float* Mutable pointer, or nullptr if not mutable/not FP32.
         */
        virtual float *mutable_fp32_data() { return nullptr; }

        /**
         * @brief Returns true if this tensor can provide fp32_data() directly without dequantization.
         *
         * For FP32Tensor, TensorSlice (of FP32 inner): returns true.
         * For FP16Tensor, BF16Tensor: returns false (needs conversion).
         * For quantized weight tensors: returns false (needs dequantization).
         *
         * Use this to check if fp32_data() is zero-cost or requires conversion.
         */
        virtual bool is_fp32_backed() const { return false; }

        // Device transfers (Phase 4.2)
        virtual bool copyFrom(const TensorBase *src) = 0; // Copy data from another tensor (handles device transfers)

        // Kernel creation (only for weight matrices - GEMM)
        // NOTE: RoPE, SwiGLU, Softmax, RMSNorm, Attention moved to IActivationTensor
        // NOTE: For row-sliced GEMM (tensor parallelism), use TensorSlice wrapper
        virtual std::unique_ptr<ITensorGemm> createGemm() = 0;

        /**
         * @brief Release raw weight data after GEMM kernel has been packed
         *
         * For quantized weight tensors (Q8_0, Q4_0, etc.), this releases the
         * original GGUF format data after the GEMM kernel has repacked it into
         * INT8 VNNI format. This cuts weight memory usage roughly in half.
         *
         * Call this after preparing GEMM weights / creating the GEMM engine to free
         * the original quantized data.
         * The tensor remains usable for GEMM operations via the cached kernel,
         * but other operations (to_fp32, decode_block_at, etc.) will fail.
         *
         * @note Default throws - each tensor type must implement explicitly
         * @note Not thread-safe - caller must ensure no concurrent access
         * @note For TensorSlice, this forwards to the inner tensor
         */
        virtual void release_raw_data()
        {
            throw std::runtime_error(
                "release_raw_data() not implemented for tensor type " +
                std::to_string(static_cast<int>(native_type())) +
                " - add implementation to free raw_data_ after GEMM packing");
        }

        /**
         * @brief Check if raw data has been released
         * @return true if raw data was released (GEMM still works via cached kernel)
         */
        virtual bool is_raw_data_released() const { return false; }

        /**
         * @brief Release ALL host-side weight data for GPU-offloaded tensors
         *
         * After weights have been uploaded to GPU, this frees:
         *  1. raw_data_ (original quantized blocks)
         *  2. dequant_cache_ (FP32 decompressed cache, if populated)
         *  3. mmap_owner_ reference (allows mmap unmap when all tensors release)
         *
         * This is more aggressive than release_raw_data() which only frees the
         * original quantized blocks. Use this when the tensor is fully on GPU
         * and no host-side operations are needed.
         *
         * @note Default calls release_raw_data(); subclasses override to clear dequant_cache_ + mmap_owner_
         */
        virtual void release_host_weight_data()
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
        }

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

        // =====================================================================
        // Tensor Comparison Utilities (for testing and debugging)
        // =====================================================================

        /**
         * @brief Compute cosine similarity between this tensor and another
         *
         * Cosine similarity measures the angle between two vectors, ignoring magnitude.
         * Result range: [-1, 1] where 1.0 = identical direction, 0.0 = orthogonal, -1.0 = opposite
         *
         * Formula: cos(θ) = (A · B) / (||A|| × ||B||)
         *
         * @param other Tensor to compare with (must have same element count)
         * @return Cosine similarity in range [-1.0, 1.0], or NaN if either tensor is zero
         * @throws std::invalid_argument if element counts don't match
         *
         * @note Uses fp32_data() for comparison, so Q8_1 tensors are dequantized
         * @note Thread-safe: reads only
         *
         * Example usage in tests:
         * @code
         * float similarity = tensor_a->cosineSimilarityTo(tensor_b);
         * EXPECT_GT(similarity, 0.999);  // Very similar
         * @endcode
         */
        double cosineSimilarityTo(const TensorBase *other) const;

        /**
         * @brief Compute maximum absolute difference between this tensor and another
         *
         * Useful for detecting worst-case numerical divergence.
         *
         * @param other Tensor to compare with (must have same element count)
         * @return max(|this[i] - other[i]|) for all i
         * @throws std::invalid_argument if element counts don't match
         *
         * @note Uses fp32_data() for comparison
         *
         * Example usage in tests:
         * @code
         * float max_diff = tensor_a->maxAbsDiffTo(tensor_b);
         * EXPECT_LT(max_diff, 1e-3f);  // Max error < 0.001
         * @endcode
         */
        float maxAbsDiffTo(const TensorBase *other) const;

        /**
         * @brief Compute mean absolute difference between this tensor and another
         *
         * Useful for measuring average numerical divergence.
         *
         * @param other Tensor to compare with (must have same element count)
         * @return mean(|this[i] - other[i]|) for all i
         * @throws std::invalid_argument if element counts don't match
         *
         * @note Uses fp32_data() for comparison
         *
         * Example usage in tests:
         * @code
         * float mean_diff = tensor_a->meanAbsDiffTo(tensor_b);
         * EXPECT_LT(mean_diff, 1e-5f);  // Mean error < 0.00001
         * @endcode
         */
        float meanAbsDiffTo(const TensorBase *other) const;

        /**
         * @brief Compute relative L2 norm (Frobenius distance normalized by reference)
         *
         * Measures relative error as the ratio of difference norm to reference norm.
         * Formula: ||A - B||_2 / ||B||_2
         *
         * @param other Reference tensor to compare against (must have same element count)
         * @return Relative L2 norm (0.0 = identical, larger = more different)
         * @throws std::invalid_argument if element counts don't match
         *
         * @note Uses fp32_data() for comparison
         * @note Returns INFINITY if other tensor is zero
         *
         * Example usage in tests:
         * @code
         * double rel_l2 = tensor_a->relativeL2To(tensor_b);
         * EXPECT_LT(rel_l2, 0.01);  // Relative error < 1%
         * @endcode
         */
        double relativeL2To(const TensorBase *other) const;

        /**
         * @brief Compute KL divergence treating tensors as probability distributions
         *
         * KL divergence measures how one probability distribution differs from another.
         * This method applies softmax to both tensors before computing divergence,
         * making it suitable for comparing logits.
         *
         * Formula: KL(P || Q) = Σ P(i) × log(P(i) / Q(i))
         *
         * @param other Reference distribution (Q) to compare against
         * @return KL divergence (≥0, lower = more similar), or INFINITY if distributions incompatible
         * @throws std::invalid_argument if element counts don't match
         *
         * @note Applies softmax to convert logits to probabilities
         * @note Uses fp32_data() for comparison
         * @note KL divergence is asymmetric: KL(A||B) ≠ KL(B||A)
         *
         * Example usage for comparing logits:
         * @code
         * double kl_div = logits_a->klDivergenceTo(logits_b);
         * EXPECT_LT(kl_div, 0.1);  // Distributions are similar
         * @endcode
         */
        double klDivergenceTo(const TensorBase *other) const;

        /**
         * @brief Get a summary of comparison metrics between this tensor and another
         *
         * Computes all comparison metrics in one call for efficiency.
         * Returns a struct with: cosine_similarity, max_abs_diff, mean_abs_diff, relative_l2
         *
         * @param other Tensor to compare with (must have same element count)
         * @return ComparisonSummary struct with all metrics
         * @throws std::invalid_argument if element counts don't match
         *
         * @note Does NOT compute KL divergence (expensive softmax)
         */
        struct ComparisonSummary
        {
            double cosine_similarity; ///< Cosine similarity [-1, 1]
            float max_abs_diff;       ///< Maximum absolute difference
            float mean_abs_diff;      ///< Mean absolute difference
            double relative_l2;       ///< Relative L2 norm

            std::string toString() const
            {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                              "ComparisonSummary{cos=%.6f, max=%.6e, mean=%.6e, rel_l2=%.6e}",
                              cosine_similarity, max_abs_diff, mean_abs_diff, relative_l2);
                return buf;
            }
        };

        ComparisonSummary compareTo(const TensorBase *other) const;

        // =========================================================================
        // Debug-Mode Validity Tracking (Task 4: Tensor Validity Assertions)
        // =========================================================================
        //
        // In debug builds, tracks tensor validity state to catch use-after-move,
        // use-after-free, and use-after-pool-release bugs. Zero cost in release builds.
        //
        // Usage:
        //   tensor->invalidate("Released back to buffer pool");
        //   tensor->data();  // Will trigger assertion with clear error message
        //
        // Enabled only when NDEBUG is NOT defined (debug builds).

#ifndef NDEBUG
        /**
         * @brief Check if tensor is currently valid for use
         * @return true if tensor data can be safely accessed
         */
        bool isValid() const { return valid_; }

        /**
         * @brief Mark this tensor as invalid (debug builds only)
         *
         * Call this when:
         * - Tensor is returned to a buffer pool
         * - Tensor data is moved/stolen
         * - Tensor is aliased to another buffer
         *
         * @param reason Human-readable explanation (shown in assertion)
         */
        void invalidate(const std::string &reason)
        {
            valid_ = false;
            invalidation_reason_ = reason;
        }

        /**
         * @brief Re-validate tensor after reallocation (debug builds only)
         *
         * Call this when:
         * - Buffer pool allocates this tensor for new use
         * - Data is restored after temporary invalidation
         */
        void revalidate()
        {
            valid_ = true;
            invalidation_reason_.clear();
        }

        /**
         * @brief Get the reason this tensor was invalidated
         * @return Reason string, or empty if tensor is valid
         */
        const std::string &invalidationReason() const { return invalidation_reason_; }

#else
        // No-op in release builds - zero overhead
        bool isValid() const { return true; }
        void invalidate(const std::string &) {}
        void revalidate() {}
        const std::string &invalidationReason() const
        {
            static std::string empty;
            return empty;
        }
#endif

        // =========================================================================
        // Debug Name (for better error messages)
        // =========================================================================

        /**
         * @brief Set a human-readable name for this tensor (debug builds)
         * @param name Descriptive name like "layer0_attention_q" or "ffn_gate"
         */
        void setDebugName(const std::string &name) { debug_name_ = name; }

        /**
         * @brief Get the debug name for this tensor
         * @return Debug name, or empty string if not set
         */
        const std::string &debugName() const { return debug_name_; }

        // =========================================================================
        // Tensor Layout Contract (Phase 3: Tensor Layout Contracts)
        // =========================================================================

        /**
         * @brief Get the memory layout of this tensor
         *
         * Layouts define how multi-dimensional data is organized in memory.
         * For attention tensors, this indicates whether data is sequence-major,
         * head-major, position-major, etc.
         *
         * @return Current layout, or TensorLayout::UNKNOWN if not set
         * @see TensorLayout.h for layout definitions
         */
        virtual TensorLayout layout() const { return layout_; }

        /**
         * @brief Set the memory layout of this tensor
         *
         * Call this when:
         * - Creating Q/K/V tensors with specific layouts
         * - After transposing tensor data
         * - When reinterpreting tensor shape semantics
         *
         * @param layout New layout to set
         * @note This only changes metadata, not the actual data layout.
         *       Use transpose_to() for actual data reorganization.
         */
        virtual void setLayout(TensorLayout layout) { layout_ = layout; }

        // =========================================================================
        // Backend Dependency Injection (for testing)
        // =========================================================================

        /**
         * @brief Inject a custom backend for this tensor (test-only)
         *
         * When set, all backend operations (ensureOnDevice, ensureOnHost, etc.)
         * use this backend instead of the global CUDA/ROCm backend lookup.
         * This enables unit testing of coherence logic with MockBackend.
         *
         * @param backend Pointer to IBackend implementation (caller retains ownership)
         * @note Non-owning: caller must ensure backend outlives the tensor
         * @note Thread-safe: acquires coherence_mutex_
         */
        void setBackendForTesting(IBackend *backend)
        {
            std::lock_guard<std::mutex> lock(coherence_mutex_);
            injected_backend_ = backend;
        }

        /**
         * @brief Clear the injected backend, reverting to global backend lookup
         */
        void clearBackendForTesting()
        {
            std::lock_guard<std::mutex> lock(coherence_mutex_);
            injected_backend_ = nullptr;
        }

    protected:
        // Default constructor for derived classes
        TensorBase() = default;

#ifndef NDEBUG
        // Debug validity tracking state
        bool valid_ = true;
        std::string invalidation_reason_;
#endif
        // Debug name for error messages (always present, minimal overhead)
        std::string debug_name_;

        // Tensor layout contract (Phase 3)
        TensorLayout layout_ = TensorLayout::UNKNOWN;

        // ===== Debug Validity Assertion Helper =====
#ifndef NDEBUG
        /**
         * @brief Assert tensor is valid before access (debug builds only)
         *
         * Call at the start of data(), mutable_data(), fp32_data(), and other
         * data access methods. Provides clear error message on invalid access.
         *
         * @param method_name Name of the method being called (for error message)
         * @throws std::runtime_error if tensor is invalid
         */
        void assertValid(const char *method_name) const
        {
            if (!valid_)
            {
                std::string msg = "Tensor validity assertion failed in ";
                msg += method_name;
                msg += "(): tensor was invalidated";
                if (!invalidation_reason_.empty())
                {
                    msg += " - reason: " + invalidation_reason_;
                }
                if (!debug_name_.empty())
                {
                    msg += " [tensor: " + debug_name_ + "]";
                }
                throw std::runtime_error(msg);
            }
        }
#else
        // No-op in release builds
        void assertValid(const char *) const {}
#endif

        // ===== Host/Device Memory Coherence State =====
        // Two-flag model for tracking data validity:
        //   host_valid_   = true:  Host data is current (can be read without sync)
        //   device_valid_ = true:  Device data is current (can be read without sync)
        //
        // Valid state combinations:
        //   HOST_AUTHORITATIVE:   host_valid_=true,  device_valid_=false
        //   DEVICE_AUTHORITATIVE: host_valid_=false, device_valid_=true
        //   SYNCED:               host_valid_=true,  device_valid_=true
        //   INVALID:              host_valid_=false, device_valid_=false (ERROR STATE)
        //
        // See docs/v2/TENSOR_MEMORY_COHERENCE_DESIGN.md for full design.

        void *gpu_data_ptr_ = nullptr; // GPU buffer pointer (nullptr = not on GPU)
        bool host_valid_ = true;       // Host data is current (starts true - data created on host)
        bool device_valid_ = false;    // Device data is current (starts false - no GPU alloc)

        // Injected backend for testing (non-owning). When non-null, resolveBackend()
        // returns this instead of looking up the global CUDA/ROCm backend.
        IBackend *injected_backend_ = nullptr;

        /**
         * @brief Resolve the backend for a given device
         *
         * Returns injected_backend_ if set (for testing), otherwise
         * falls back to global CUDA/ROCm backend lookup.
         */
        IBackend *resolveBackend(DeviceId device) const;

        std::optional<DeviceId> gpu_device_;      // Which GPU device (nullopt = not on GPU)
        void *device_completion_event_ = nullptr; // Event marking last kernel write (for fine-grained sync)
        std::optional<DeviceId> event_device_;    // Device where device_completion_event_ was created

        // ===== Multi-Device Coherence Tracking =====
        // Tracks which device has authoritative (current) data.
        // - nullopt: Host is authoritative (default, backward compatible)
        // - DeviceId: That GPU device is authoritative
        // This enables direct GPU-to-GPU transfers without host staging.
        std::optional<DeviceId> authoritative_device_;

        // ===== Multi-Device Buffer Management =====

        // Serializes host/device coherence and primary/secondary buffer transitions.
        // Required for LOCAL TP where the same tensor can be touched from multiple
        // device worker threads.
        mutable std::mutex coherence_mutex_;

        /**
         * @brief Map of secondary device buffers (for multi-device scenarios)
         *
         * When a tensor has been on multiple GPUs, we keep the buffers around
         * for potential reuse. Key is device type + ordinal packed into int.
         *
         * The primary GPU buffer is still stored in gpu_data_ptr_ and gpu_device_.
         * This map stores buffers for other devices the tensor has visited.
         */
        std::unordered_map<int, void *> secondary_device_buffers_;

        /**
         * @brief Set of secondary buffer keys that were allocated in BAR region
         *
         * These buffers must NOT be freed with backend->free(), they're
         * automatically reclaimed when PCIeBARBackend shuts down.
         */
        std::unordered_set<int> secondary_bar_allocated_keys_;

        /**
         * @brief Pack DeviceId into int for use as map key
         */
        static int packDeviceId(DeviceId device)
        {
            return (static_cast<int>(device.type) << 16) | device.ordinal;
        }

        /**
         * @brief Get existing GPU buffer pointer for a device, or allocate new one
         *
         * Used by transferTo() and copyTo() to ensure destination buffer exists.
         *
         * @param device Target GPU device
         * @return GPU pointer or nullptr if allocation failed
         */
        void *getOrAllocateDeviceBuffer(DeviceId device);

        // ===== Zero-Copy Mapped Memory =====
        // When a tensor uses mapped memory (hipHostMallocMapped/cudaHostAllocMapped):
        // - Host and device share the SAME physical memory via PCIe BAR
        // - GPU can read/write directly without memcpy
        // - ensureOnDevice()/ensureOnHost() become no-ops
        // - Eliminates sync memcpy bottleneck (1.5-1.7x speedup in benchmarks)
        //
        // Use for ACTIVATION tensors (read once, written once per stage).
        // Do NOT use for weight tensors (need GPU cache locality).
        //
        // Derived classes should check is_mapped_ in raw_host_data_ptr() and return
        // mapped_host_ptr_ when true.
        bool is_mapped_ = false;            // True if using mapped memory
        void *mapped_device_ptr_ = nullptr; // Device-visible pointer for mapped host memory
        void *mapped_host_ptr_ = nullptr;   // Host-visible pointer for mapped memory

        // For mapped memory: tracks whether GPU has written since last sync.
        // Set to true by mark_device_dirty(), cleared by ensureOnHost() after sync.
        // This avoids redundant hipDeviceSynchronize() calls.
        bool mapped_needs_sync_ = false;

        /**
         * @brief Initialize mapped memory for this tensor (protected helper)
         *
         * Allocates GPU-visible host memory via backend->allocateMapped().
         * Sets up is_mapped_, mapped_host_ptr_, mapped_device_ptr_, and gpu_data_ptr_.
         *
         * @param bytes Size in bytes to allocate
         * @param target_device GPU device to map for
         * @return true on success, false if allocation failed
         *
         * @note Called by derived class factories (e.g., FP32Tensor::createMapped)
         */
        bool initMappedMemory(size_t bytes, DeviceId target_device);

        /**
         * @brief Free mapped memory if allocated (called by destructor)
         */
        void freeMappedMemory();

        // ===== BAR-Backed Memory (PCIeBAR cross-vendor zero-copy) =====
        // When a tensor uses BAR-backed memory:
        // - Buffer physically resides in AMD VRAM (not system DRAM)
        // - CUDA writes go directly to AMD VRAM via PCIe posted writes (~2.65 GB/s)
        // - AMD accesses locally (full VRAM bandwidth ~900 GB/s)
        // - Host access via mmap (O_SYNC, uncached, ~1-2 GB/s)
        //
        // Use for: allreduce output buffers, small communication buffers
        // Don't use for: weight tensors, large activation tensors
        //
        // See BARBackedTensor.h for detailed design documentation.
        bool is_bar_backed_ = false;          // True if using BAR-backed allocation
        size_t bar_offset_ = 0;               // Offset within BAR region
        size_t bar_size_ = 0;                 // Allocated size in BAR
        void *bar_rocm_ptr_ = nullptr;        // ROCm-accessible pointer (mmap'd BAR = AMD VRAM)
        void *bar_cuda_device_ptr_ = nullptr; // CUDA-accessible pointer (cuMemHostGetDevicePointer)
        DeviceId bar_host_device_;            // ROCm device whose BAR hosts this buffer
        DeviceId bar_accessor_device_;        // CUDA device with registered BAR access

        // ===== HIP Staging Buffer for BAR-Backed Tensors =====
        // CRITICAL FINDING: HIP kernels CANNOT directly dereference BAR mmap addresses
        // (causes memory access fault). However, hipMemcpy(D2D) with BAR as destination
        // WORKS at ~4+ GB/s using the AMD DMA engine.
        //
        // Architecture for BAR-backed ROCm tensors:
        //   1. ROCm kernel writes to hip_staging_ptr_ (real HIP device memory)
        //   2. After kernel, copy: hipMemcpy(D2D, bar_rocm_ptr_, hip_staging_ptr_, size)
        //   3. CUDA reads from bar_cuda_device_ptr_ (BAR)
        //
        // This adds ~1-2μs overhead for the D2D copy but uses full DMA bandwidth.
        void *hip_staging_ptr_ = nullptr; // HIP device buffer for kernel writes (hipMalloc)
        bool owns_hip_staging_ = false;   // True if we need to hipFree this

        // Forward declaration - PCIeBARBackend manages BAR allocations
        // Pointer stored here for deallocation in destructor
        class PCIeBARBackend *bar_backend_ = nullptr;

        /**
         * @brief Initialize BAR-backed memory for this tensor (protected helper)
         *
         * Allocates buffer within the BAR-mapped region of an AMD GPU's VRAM.
         * The buffer is accessible by both CUDA (via registered IOMEMORY pointer)
         * and ROCm (via direct local VRAM access).
         *
         * @param bytes Size in bytes to allocate
         * @param cuda_device CUDA device that will write to this tensor
         * @param rocm_device ROCm device whose BAR hosts this tensor
         * @param backend PCIeBARBackend managing the BAR (must outlive tensor)
         * @return true on success, false if allocation failed
         *
         * @note Called by derived class factories (e.g., FP32Tensor::createBARBacked)
         */
        bool initBARBackedMemory(size_t bytes, DeviceId cuda_device, DeviceId rocm_device,
                                 class PCIeBARBackend *backend);

        /**
         * @brief Free BAR-backed memory if allocated (called by destructor)
         */
        void freeBARBackedMemory();

        // ===== Pinned Memory for Fast GPU Transfers =====
        // When host buffer is pinned (page-locked), hipMemcpy/cudaMemcpy can:
        // - Use DMA transfers instead of staging through internal buffers
        // - Run truly asynchronously without blocking on device sync
        // Host memory is pinned lazily on first GPU transfer attempt.
        bool host_pinned_ = false;        // True if host buffer is registered as pinned
        size_t pinned_bytes_ = 0;         // Size of pinned region (for unregister)
        void *pinned_host_ptr_ = nullptr; // Pointer used when pinning (for unpin in destructor)

        /**
         * @brief Register host buffer as pinned memory for fast GPU transfers
         *
         * Called automatically by ensureOnDevice() on first GPU transfer.
         * Uses hipHostRegister/cudaHostRegister to pin existing pageable memory.
         *
         * @return true if pinning succeeded (or already pinned), false on error
         * @note Safe to call multiple times - will return true if already pinned
         */
        bool ensureHostPinned();

        /**
         * @brief Unregister host buffer from pinned memory
         *
         * Called automatically by destructor or when releasing GPU resources.
         * Safe to call even if not pinned (no-op).
         */
        void unpinHostMemory();

        // ===== Abstract Accessors for Lazy Transfer =====
        // Each tensor type can override these for GPU support.
        // Default implementations throw - tensors that need GPU must override.
        // TensorBase uses them in ensureOnDevice/ensureOnHost.

        /**
         * @brief Get raw pointer to host data buffer
         * @return Pointer to start of host data (casted from internal storage type)
         * @note For FP32Tensor: returns host_data_.data()
         * @note For quantized: returns blocks_.data()
         * @note Default throws - override in tensor types that support GPU transfer
         */
        virtual void *raw_host_data_ptr();
        virtual const void *raw_host_data_ptr() const;

        /**
         * @brief Get total byte size of tensor data
         * @return Number of bytes to transfer to/from GPU
         * @note For FP32Tensor: element_count() * sizeof(float)
         * @note For quantized: num_blocks * sizeof(BlockType)
         * @note Default throws - override in tensor types that support GPU transfer
         */
        virtual size_t byte_size() const;

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
     * - `TypedTensorBase<FP32Tensor, float>`: Type-safe `typed_data()` returning `float*`
     * - `TensorBase`: Core tensor API (device affinity, shape, lazy transfers)
     * - `IActivationTensor`: Kernel creation, in-place ops (RMSNorm, RoPE, SwiGLU)
     * - `ITensorGemmTileDataProvider`: Block-wise FP32 decode (block_size=32, identity decode)
     * - `IQ8_0Decodable`: On-the-fly quantization to Q8_0 (for testing/compatibility)
     * - `IQ8_1Decodable`: On-the-fly quantization to Q8_1 (for INT8 GEMM activations)
     *
     * **Usage**: Default activation tensor for FP32 inference pipelines. Provides baseline
     * numerical accuracy for parity testing against PyTorch/llama.cpp.
     *
     * **Type-Safe Access**: Use `typed_data()` for native `float*` access. The legacy
     * `data()` method is equivalent for FP32Tensor.
     */
    class FP32Tensor : public TypedTensorBase<FP32Tensor, float>,
                       public TensorBase,
                       public IActivationTensor,
                       public ITensorGemmTileDataProvider,
                       public IQ8_0Decodable,
                       public IQ8_1Decodable
    {
    public:
        /// Native storage type (same as TypedTensorBase::value_type)
        using value_type = float;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::FP32; }

        // ===== CRTP Implementation for TypedTensorBase =====
        /// Called by TypedTensorBase::typed_data()
        const float *data_impl() const
        {
            if (is_mapped_ && mapped_host_ptr_)
                return static_cast<const float *>(mapped_host_ptr_);
            return host_data_.data();
        }
        /// Called by TypedTensorBase::mutable_typed_data()
        float *mutable_data_impl()
        {
            if (is_mapped_ && mapped_host_ptr_)
                return static_cast<float *>(mapped_host_ptr_);
            return host_data_.data();
        }

        /// @brief Construct FP32Tensor on specified device
        /// @param shape Tensor dimensions
        /// @param device Device placement (default: CPU)
        explicit FP32Tensor(const std::vector<size_t> &shape, DeviceId device = DeviceId::cpu());

        /**
         * @brief Create a zero-copy mapped FP32Tensor for GPU execution
         *
         * Allocates tensor data in mapped host memory that is directly accessible
         * by the GPU via PCIe. This eliminates memcpy overhead:
         * - ensureOnDevice() becomes a no-op (GPU reads directly from host memory)
         * - ensureOnHost() becomes a no-op (host reads directly, no D2H copy)
         * - 1.5x-1.7x faster than traditional H2D/D2H memcpy pattern
         *
         * **Use for**: Activation tensors (read once, written once per stage)
         * **Don't use for**: Weight tensors (need GPU cache locality for repeated reads)
         *
         * @param shape Tensor dimensions
         * @param target_device GPU device for mapped memory (must be CUDA or ROCm)
         * @return unique_ptr<FP32Tensor> with mapped memory, or nullptr on failure
         *
         * @note Falls back to regular allocation if mapped memory unavailable
         * @note Currently supported: ROCm (hipHostMallocMapped)
         */
        static std::unique_ptr<FP32Tensor> createMapped(
            const std::vector<size_t> &shape,
            DeviceId target_device);

        /**
         * @brief Create a BAR-backed FP32Tensor for zero-copy PCIeBAR allreduce
         *
         * Allocates tensor data in AMD GPU VRAM via PCIe BAR mapping.
         * CUDA kernels can write directly to this buffer via the BAR,
         * and AMD kernels access it as local VRAM.
         *
         * **Memory model**:
         * - Buffer physically resides in AMD VRAM (not system DRAM)
         * - CUDA writes via PCIe posted writes (~2.65 GB/s on PCIe 3.0)
         * - AMD accesses locally (full VRAM bandwidth ~900 GB/s)
         * - Host access via mmap (slow, uncached ~1-2 GB/s)
         *
         * **Coherence model**:
         * - Both CUDA and ROCm see the same physical memory
         * - mark_device_dirty() sets flag, no copy needed
         * - ensureOnDevice() is no-op for both CUDA and ROCm
         *
         * **Use for**: allreduce output buffers, small communication buffers
         * **Don't use for**: weight tensors, large activation tensors
         *
         * @param shape Tensor dimensions
         * @param cuda_device CUDA device that will write to this tensor
         * @param rocm_device ROCm device whose BAR will host this tensor
         * @param backend PCIeBARBackend managing the BAR (must outlive tensor)
         * @return unique_ptr<FP32Tensor> with BAR-backed memory, or nullptr on failure
         *
         * @note Requires HAVE_CUDA && HAVE_ROCM
         * @note Requires CAP_SYS_ADMIN for IOMEMORY registration
         * @see BARBackedTensor.h for detailed design documentation
         */
        static std::unique_ptr<FP32Tensor> createBARBacked(
            const std::vector<size_t> &shape,
            DeviceId cuda_device,
            DeviceId rocm_device,
            class PCIeBARBackend *backend);

        ~FP32Tensor() override;

        // ===== TensorBase Interface =====
        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::FP32; }

        DeviceId home_device() const override { return device_; }

        const float *data() const override;
        float *mutable_data() override;

        // ===== Diamond Inheritance Resolution =====
        // Both TypedTensorBase and TensorBase provide ITensor overrides.
        // FP32Tensor must provide final overrides to disambiguate.
        // We delegate to TensorBase's implementations (already working).
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        // ===== Unified FP32 Access (Phase 1 Infrastructure) =====
        float *mutable_fp32_data() override { return mutable_data(); }
        bool is_fp32_backed() const override { return true; }

        bool copyFrom(const TensorBase *src) override;

        std::unique_ptr<ITensorGemm> createGemm() override;

        // ===== IActivationTensor Interface =====
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;
        std::unique_ptr<ITensorAttention> createAttention() override;
        std::unique_ptr<ITensorEmbedding> createEmbedding() override;
        ActivationPack to_int8_activation_pack(int rows, int cols) const override;
        bool quantize_to_q8_1(void *q8_1_buffer, int m, int k) const override;

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

        bool applyRMSNorm(
            const float *gamma,
            int seq_len,
            int d_model,
            float eps = 1e-6f,
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

        // ===== Host data release =====
        void release_raw_data() override
        {
            if (!is_view_ && !is_mapped_)
            {
                AlignedVector<float> tmp;
                std::swap(host_data_, tmp);
            }
            raw_data_released_ = true;
        }

        bool is_raw_data_released() const override { return raw_data_released_; }

        void release_host_weight_data() override
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
        }

    protected:
        // ===== Lazy Transfer Accessors (Phase 1) =====
        void *raw_host_data_ptr() override;
        const void *raw_host_data_ptr() const override;
        size_t byte_size() const override { return element_count() * sizeof(float); }

    private:
        // Private constructor for creating views
        FP32Tensor(const std::vector<size_t> &shape,
                   DeviceId device,
                   AlignedVector<float> *parent_data,
                   size_t data_offset,
                   std::shared_ptr<FP32Tensor> parent);

        std::vector<size_t> shape_;
        DeviceId device_; // Type-safe device identification

        // Ownership model:
        // - If is_view_ == false && !is_mapped_: owns host_data_ (64-byte aligned for SIMD)
        // - If is_view_ == true: parent_data_ptr_ points to parent's host_data_
        // - If is_mapped_ == true: uses mapped_host_data_ (allocated by backend)
        bool is_view_;
        AlignedVector<float> host_data_;        // Owned data (64-byte aligned, only used when !is_view_ && !is_mapped_)
        AlignedVector<float> *parent_data_ptr_; // Borrowed data pointer (only used when is_view_)
        size_t view_offset_;                    // Offset into parent data (only used when is_view_)
        std::shared_ptr<FP32Tensor> parent_;    // Keep parent alive (only used when is_view_)
        bool raw_data_released_ = false;        // Track if host data has been released
        // Note: mapped memory uses TensorBase::mapped_host_ptr_ (cast to float* in raw_host_data_ptr)
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
    class FP16Tensor : public TypedTensorBase<FP16Tensor, uint16_t>,
                       public TensorBase,
                       public IActivationTensor,
                       public ITensorGemmTileDataProvider,
                       public IQ8_0Decodable,
                       public IQ8_1Decodable
    {
    public:
        /// Native storage type (same as TypedTensorBase::value_type)
        using value_type = uint16_t;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::FP16; }

        // ===== CRTP Implementation for TypedTensorBase =====
        /// Called by TypedTensorBase::typed_data() - returns FP16 (uint16_t*)
        const uint16_t *data_impl() const { return fp16_data(); }
        /// Called by TypedTensorBase::mutable_typed_data()
        uint16_t *mutable_data_impl() { return mutable_fp16_data(); }

        explicit FP16Tensor(const std::vector<size_t> &shape);
        FP16Tensor(const std::vector<size_t> &shape, const std::vector<uint16_t> &fp16_data);
        ~FP16Tensor() override;

        // TensorBase interface
        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::FP16; }

        DeviceId home_device() const override { return device_; }

        const float *data() const override; // Dequantizes to cache
        float *mutable_data() override;     // Not supported

        // ===== Diamond Inheritance Resolution =====
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        // ===== Unified FP32 Access (Phase 1 Infrastructure) =====
        // FP16 is NOT FP32-backed - requires FP16→FP32 conversion
        bool is_fp32_backed() const override { return false; }

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Memory management - FP16 doesn't have separate raw data to release
        void release_raw_data() override { /* no-op: FP16 has no separate raw block data */ }
        bool is_raw_data_released() const override { return false; /* FP16 always has its data */ }

        void release_host_weight_data() override
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
            {
                decltype(dequant_cache_) tmp;
                std::swap(dequant_cache_, tmp);
            }
        }

        // IActivationTensor interface - activation-only operations
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;
        std::unique_ptr<ITensorAttention> createAttention() override;
        std::unique_ptr<ITensorEmbedding> createEmbedding() override;
        ActivationPack to_int8_activation_pack(int rows, int cols) const override;
        bool quantize_to_q8_1(void *q8_1_buffer, int m, int k) const override;

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

        bool applyRMSNorm(
            const float *gamma,
            int seq_len,
            int d_model,
            float eps = 1e-6f,
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
            dequant_cache_.clear();
            if (is_view_ && parent_)
            {
                parent_->dequant_cache_.clear();
            }
            return is_view_ ? (parent_data_ptr_->data() + view_offset_) : host_fp16_data_.data();
        }
        void from_fp32(const float *fp32_data, size_t count);
        void to_fp32(float *fp32_data, size_t count) const;

    protected:
        // ===== Lazy Transfer Accessors (Phase 3) =====
        void *raw_host_data_ptr() override;
        const void *raw_host_data_ptr() const override;
        size_t byte_size() const override { return element_count() * sizeof(uint16_t); }

    private:
        // Private constructor for creating views
        FP16Tensor(const std::vector<size_t> &shape,
                   DeviceId device,
                   AlignedVector<uint16_t> *parent_data,
                   size_t data_offset,
                   std::shared_ptr<FP16Tensor> parent);

        std::vector<size_t> shape_;
        DeviceId device_; // Type-safe device identification

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
    class BF16Tensor : public TypedTensorBase<BF16Tensor, uint16_t>,
                       public TensorBase,
                       public IActivationTensor,
                       public ITensorGemmTileDataProvider,
                       public IQ8_0Decodable,
                       public IQ8_1Decodable
    {
    public:
        /// Native storage type (same as TypedTensorBase::value_type)
        using value_type = uint16_t;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::BF16; }

        // ===== CRTP Implementation for TypedTensorBase =====
        /// Called by TypedTensorBase::typed_data() - returns BF16 (uint16_t*)
        const uint16_t *data_impl() const { return bf16_data(); }
        /// Called by TypedTensorBase::mutable_typed_data()
        uint16_t *mutable_data_impl() { return mutable_bf16_data(); }

        explicit BF16Tensor(const std::vector<size_t> &shape);
        BF16Tensor(const std::vector<size_t> &shape, const std::vector<uint16_t> &bf16_data);
        ~BF16Tensor() override;

        // TensorBase interface
        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::BF16; }

        DeviceId home_device() const override { return device_; }

        const float *data() const override; // Dequantizes to cache
        float *mutable_data() override;     // Not supported

        // ===== Diamond Inheritance Resolution =====
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Memory management - BF16 doesn't have separate raw data to release
        void release_raw_data() override { /* no-op: BF16 has no separate raw block data */ }
        bool is_raw_data_released() const override { return false; /* BF16 always has its data */ }

        void release_host_weight_data() override
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
            {
                decltype(dequant_cache_) tmp;
                std::swap(dequant_cache_, tmp);
            }
        }

        // IActivationTensor interface - activation-only operations
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;
        std::unique_ptr<ITensorAttention> createAttention() override;
        std::unique_ptr<ITensorEmbedding> createEmbedding() override;
        ActivationPack to_int8_activation_pack(int rows, int cols) const override;
        bool quantize_to_q8_1(void *q8_1_buffer, int m, int k) const override;

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

        bool applyRMSNorm(
            const float *gamma,
            int seq_len,
            int d_model,
            float eps = 1e-6f,
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

        // Unified Data Access Interface - BF16 requires conversion (not FP32-backed)
        bool is_fp32_backed() const override { return false; }

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

    protected:
        // ===== Lazy Transfer Accessors (Phase 3) =====
        void *raw_host_data_ptr() override;
        const void *raw_host_data_ptr() const override;
        size_t byte_size() const override { return element_count() * sizeof(uint16_t); }

    private:
        // Private constructor for creating views
        BF16Tensor(const std::vector<size_t> &shape,
                   DeviceId device,
                   AlignedVector<uint16_t> *parent_data,
                   size_t data_offset,
                   std::shared_ptr<BF16Tensor> parent);

        std::vector<size_t> shape_;
        DeviceId device_; // Type-safe device identification

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
    class INT8Tensor : public TypedTensorBase<INT8Tensor, int8_t>,
                       public TensorBase,
                       public ITensorGemmTileDataProvider
    {
    public:
        /// Native storage type (same as TypedTensorBase::value_type)
        using value_type = int8_t;

        // ===== CRTP Implementation for TypedTensorBase =====
        /// Called by TypedTensorBase::typed_data() - returns int8_t*
        const int8_t *data_impl() const { return host_int8_data_.data(); }
        /// Called by TypedTensorBase::mutable_typed_data()
        int8_t *mutable_data_impl() { return host_int8_data_.data(); }

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::INT8; }

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

        DeviceId home_device() const override { return device_; }

        const float *data() const override; // Dequantizes to cache
        float *mutable_data() override;     // Not supported

        // ===== Diamond Inheritance Resolution =====
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        bool copyFrom(const TensorBase *src) override;

        // TensorBase pure virtual - required implementation
        std::unique_ptr<ITensorGemm> createGemm() override;

        // Phase 2 fused kernel factory methods
        std::unique_ptr<class FusedGEMM> createFusedDualGemm(TensorBase *gate_weight, TensorBase *up_weight);
        std::unique_ptr<class FusedGEMM> createFusedTripleGemm(TensorBase *q_weight, TensorBase *k_weight, TensorBase *v_weight);

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

        // Unified Data Access Interface - INT8 requires conversion (not FP32-backed)
        bool is_fp32_backed() const override { return false; }

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

    protected:
        // ===== Lazy Transfer Accessors (Phase 3) =====
        void *raw_host_data_ptr() override { return host_int8_data_.data(); }
        const void *raw_host_data_ptr() const override { return host_int8_data_.data(); }
        size_t byte_size() const override { return element_count() * sizeof(int8_t); }

    private:
        std::vector<size_t> shape_;
        DeviceId device_ = DeviceId::cpu();
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
    class INT32Tensor : public TypedTensorBase<INT32Tensor, int32_t>,
                        public TensorBase
    {
    public:
        /// Native storage type (same as TypedTensorBase::value_type)
        using value_type = int32_t;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::INT32; }

        // ===== CRTP Implementation for TypedTensorBase =====
        /// Called by TypedTensorBase::typed_data() - returns int32_t*
        const int32_t *data_impl() const { return host_int32_data_.data(); }
        /// Called by TypedTensorBase::mutable_typed_data()
        int32_t *mutable_data_impl() { return host_int32_data_.data(); }

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

        DeviceId home_device() const override { return device_; }

        const float *data() const override; // Dequantizes to cache
        float *mutable_data() override;     // Not supported

        // ===== Diamond Inheritance Resolution =====
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

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

        // Unified Data Access Interface - INT32 requires conversion (not FP32-backed)
        bool is_fp32_backed() const override { return false; }

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

    protected:
        // ===== Lazy Transfer Accessors (Phase 3) =====
        void *raw_host_data_ptr() override { return host_int32_data_.data(); }
        const void *raw_host_data_ptr() const override { return host_int32_data_.data(); }
        size_t byte_size() const override { return element_count() * sizeof(int32_t); }

    private:
        std::vector<size_t> shape_;
        DeviceId device_ = DeviceId::cpu();
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

    class IQ4_NLTensor : public TypedTensorBase<IQ4_NLTensor, IQ4_NLBlock>, public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable, public IINT8Unpackable
    {
    public:
        /// Native storage type for CRTP-style type-safe access
        using value_type = IQ4_NLBlock;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::IQ4_NL; }

        // TypedTensorBase CRTP implementation
        const IQ4_NLBlock *data_impl() const { return reinterpret_cast<const IQ4_NLBlock *>(raw_host_data_ptr()); }
        IQ4_NLBlock *mutable_data_impl() { throw std::runtime_error("IQ4_NLTensor: weight tensors are read-only"); }
        const IQ4_NLBlock *blocks() const { return typed_data(); }

        IQ4_NLTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        /// Zero-copy constructor for mmap-backed data (no memcpy)
        IQ4_NLTensor(const std::vector<size_t> &shape, const uint8_t *mmap_data,
                     size_t byte_size, std::shared_ptr<void> mmap_lifetime_owner);
        ~IQ4_NLTensor() override;

        // TensorBase interface
        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ4_NL; }

        DeviceId home_device() const override { return device_; }

        const float *data() const override; // Dequantizes to temp buffer
        float *mutable_data() override;     // Not supported for quantized tensors

        // Diamond inheritance resolution (ITensor implemented by both TypedTensorBase and TensorBase)
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const override { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override; // Fused dequant+GEMM
        ITensorGemm *createGemmRaw();

        // Memory management - release raw data after GEMM packing
        void release_raw_data() override
        {
            if (!is_view_ && !raw_data_released_)
            {
                raw_data_.clear();
                raw_data_.shrink_to_fit();
                raw_data_released_ = true;
            }
        }
        bool is_raw_data_released() const override { return raw_data_released_; }

        void release_host_weight_data() override
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
            {
                decltype(dequant_cache_) tmp;
                std::swap(dequant_cache_, tmp);
            }
            mmap_owner_.reset();
        }

        // IINT8Unpackable interface
        void unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const override;
        float get_block_scale(size_t row_idx, size_t k_block_offset) const override;
        float get_block_min(size_t row_idx, size_t k_block_offset) const override;

        size_t superblock_size() const override { return 256; }
        const NativeVnniFormatInfo *vnniFormatInfo() const override
        {
            static constexpr NativeVnniFormatInfo info{4, 16, false, false, false, 127.0f};
            return &info;
        }
        void packVnniBlock(const VnniPackContext &ctx, int n, int b) const override;

        void unpack_superblock_to_int8(
            size_t row_idx,
            size_t superblock_idx,
            int8_t *output,
            float *scales = nullptr,
            float *mins = nullptr) const override
        {
            const size_t blocks_per_row = (shape_[1] + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock *>(data_ptr);
            const size_t base_block = superblock_idx * 8;
            const IQ4_NLBlock *src = &blocks[row_idx * blocks_per_row + base_block];

            // Unpack 8 consecutive blocks (8 × 32 = 256 elements)
            for (int i = 0; i < 8; ++i)
            {
                simd::unpack_iq4_nl_to_int8(src[i], output + i * 32);
            }

            if (scales)
            {
                for (int i = 0; i < 8; ++i)
                {
                    scales[i] = fp16_to_fp32(src[i].d);
                }
            }
            if (mins)
            {
                for (int i = 0; i < 8; ++i)
                {
                    mins[i] = 0.0f; // IQ4_NL is symmetric
                }
            }
        }

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

        // Raw data access (raw_blocks provides access via TensorBase pattern)
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

    protected:
        // ===== Lazy Transfer Accessors (Phase 3) =====
        void *raw_host_data_ptr() override
        {
            return is_view_ ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        const void *raw_host_data_ptr() const override
        {
            return is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        size_t byte_size() const override { return data_byte_size_ > 0 ? data_byte_size_ : raw_data_.size(); }

    private:
        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)
        std::shared_ptr<void> mmap_owner_;   // Keeps mmap region alive for zero-copy tensors
        size_t data_byte_size_ = 0;          // Byte size for mmap-backed tensors (raw_data_ is empty)

        DeviceId device_;
        void *device_blocks_; // Quantized blocks on device (if uploaded)

        mutable std::vector<float> dequant_cache_; // Temporary for data() calls
        bool raw_data_released_ = false;           // Track if raw data was released after GEMM pack

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
    class Q8_0Tensor : public TypedTensorBase<Q8_0Tensor, Q8_0Block>, public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable, public IINT8Unpackable
    {
    public:
        /// Native storage type for CRTP-style type-safe access
        using value_type = Q8_0Block;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::Q8_0; }

        // TypedTensorBase CRTP implementation
        const Q8_0Block *data_impl() const { return reinterpret_cast<const Q8_0Block *>(raw_host_data_ptr()); }
        Q8_0Block *mutable_data_impl() { throw std::runtime_error("Q8_0Tensor: weight tensors are read-only"); }
        const Q8_0Block *blocks() const { return typed_data(); }

        Q8_0Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        /// Zero-copy constructor for mmap-backed data (no memcpy)
        Q8_0Tensor(const std::vector<size_t> &shape, const uint8_t *mmap_data,
                   size_t byte_size, std::shared_ptr<void> mmap_lifetime_owner);
        ~Q8_0Tensor() override;

        // TensorBase interface
        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q8_0; }

        DeviceId home_device() const override { return device_; }

        const float *data() const override;
        float *mutable_data() override;

        // Diamond inheritance resolution (ITensor implemented by both TypedTensorBase and TensorBase)
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Memory management - release raw data after GEMM packing
        void release_raw_data() override
        {
            if (!is_view_ && !raw_data_released_)
            {
                raw_data_.clear();
                raw_data_.shrink_to_fit();
                raw_data_released_ = true;
            }
        }
        bool is_raw_data_released() const override { return raw_data_released_; }

        void release_host_weight_data() override
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
            {
                decltype(dequant_cache_) tmp;
                std::swap(dequant_cache_, tmp);
            }
            mmap_owner_.reset();
        }

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

        // IINT8Unpackable interface
        void unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const override
        {
            const Q8_0Block *block = static_cast<const Q8_0Block *>(get_raw_block_at(row_idx, k_block_offset));
            std::memcpy(output, block->qs, 32);
        }

        float get_block_scale(size_t row_idx, size_t k_block_offset) const override
        {
            const Q8_0Block *block = static_cast<const Q8_0Block *>(get_raw_block_at(row_idx, k_block_offset));
            return fp16_to_fp32(block->d);
        }

        size_t superblock_size() const override { return 256; }

        /// Efficient override: reads Q8_0 blocks directly via typed_data()
        /// without virtual dispatch per block.
        float requantizeRowToInt8(size_t row_idx, size_t K, int8_t *output) const override;

        void unpack_superblock_to_int8(
            size_t row_idx,
            size_t superblock_idx,
            int8_t *output,
            float *scales = nullptr,
            float *mins = nullptr) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q8_0Block *blocks = reinterpret_cast<const Q8_0Block *>(data_ptr);
            const size_t base_block = superblock_idx * 8;

            // Copy 8 consecutive blocks (8 × 32 = 256 bytes)
            const Q8_0Block *src = &blocks[row_idx * blocks_per_row + base_block];
#if defined(__AVX512F__)
            // 4 × 64-byte stores = 256 bytes
            for (int i = 0; i < 4; ++i)
            {
                __m512i data = _mm512_loadu_si512(reinterpret_cast<const char *>(src) + i * 64 + 2); // Skip 2-byte scale per block pair
                // Note: Q8_0Block is 34 bytes (2 scale + 32 data), so blocks aren't contiguous
                // Fall back to per-block copy
            }
#endif
            // Blocks aren't contiguous in memory (each has a 2-byte scale header)
            // So we copy each block's qs array separately but in one function call
            for (int i = 0; i < 8; ++i)
            {
                std::memcpy(output + i * 32, src[i].qs, 32);
            }

            if (scales)
            {
                for (int i = 0; i < 8; ++i)
                {
                    scales[i] = fp16_to_fp32(src[i].d);
                }
            }
            if (mins)
            {
                for (int i = 0; i < 8; ++i)
                {
                    mins[i] = 0.0f; // Q8_0 is symmetric
                }
            }
        }

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

    protected:
        // ===== Lazy Transfer Accessors (Phase 3) =====
        void *raw_host_data_ptr() override
        {
            return is_view_ ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        const void *raw_host_data_ptr() const override
        {
            return is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        size_t byte_size() const override { return data_byte_size_ > 0 ? data_byte_size_ : raw_data_.size(); }

    private:
        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)
        std::shared_ptr<void> mmap_owner_;   // Keeps mmap region alive for zero-copy tensors
        size_t data_byte_size_ = 0;          // Byte size for mmap-backed tensors (raw_data_ is empty)

        DeviceId device_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        bool raw_data_released_ = false; // Track if raw data was released after GEMM pack

        // Private view constructor
        Q8_0Tensor(const std::vector<size_t> &shape,
                   const uint8_t *parent_raw_data,
                   size_t byte_offset,
                   std::shared_ptr<TensorBase> parent);
    };

    // ===== Q8_1 Tensor (8-bit quantization with pre-computed sum) =====

    // Implementation: Q8_1Tensor.cpp
    /**
     * @brief Q8_1 quantized tensor - ACTIVATION-ONLY format for memory bandwidth optimization
     *
     * Block format: 32 elements per block, FP16 scale + INT16 sum + int8[32] values
     * Compression: ~3.5× vs FP32 (36 bytes per 32 elements vs 128 bytes)
     *
     * IMPORTANT: Q8_1 is strictly an activation format, NEVER for weights.
     * There are no models stored in Q8_1 format - it exists purely to reduce
     * memory bandwidth during inference by keeping activations compressed.
     *
     * Design principles:
     * - Kernels read/write Q8_1 blocks directly via q8_1_blocks()/mutable_q8_1_blocks()
     * - mutable_data() THROWS - never expose FP32 for writes (defeats bandwidth savings)
     * - data() returns lazy dequantized FP32 for debugging/snapshots only
     * - Typed kernels (CPURMSNormKernelT<Q8_1>, etc.) operate directly on blocks
     *
     * Q8_1 block enables "quantize once, use many times" pattern:
     * - Pre-computes sum during quantization: sum_qs = sum(qs[i])
     * - Eliminates expensive horizontal reductions in GEMM K-loop
     * - VNNI/DP4A can use pre-computed sum for zero-point compensation
     *
     * Usage pattern:
     *   // Write to Q8_1 buffer
     *   auto output = tensor_factory->createQ8_1(shape);
     *   kernel->apply_q8_1(input->q8_1_blocks(), weights, output->mutable_q8_1_blocks(), ...);
     *
     *   // Read from Q8_1 buffer (next kernel)
     *   next_kernel->apply_q8_1(output->q8_1_blocks(), ...);
     */
    class Q8_1Tensor : public TypedTensorBase<Q8_1Tensor, Q8_1Block>,
                       public TensorBase,
                       public IActivationTensor,
                       public ITensorGemmTileDataProvider,
                       public IQ8_1Decodable,
                       public IINT8Unpackable
    {
    public:
        /// Native storage type (same as TypedTensorBase::value_type)
        using value_type = Q8_1Block;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::Q8_1; }

        // ===== CRTP Implementation for TypedTensorBase =====
        /// Called by TypedTensorBase::typed_data() - returns Q8_1Block*
        const Q8_1Block *data_impl() const { return q8_1_blocks(); }
        /// Called by TypedTensorBase::mutable_typed_data() - returns mutable Q8_1Block*
        Q8_1Block *mutable_data_impl() { return mutable_q8_1_blocks(); }

        /// Convenience alias: blocks() returns typed_data()
        const Q8_1Block *blocks() const { return typed_data(); }
        Q8_1Block *mutable_blocks() { return mutable_typed_data(); }

        /// Construct from existing raw Q8_1 block data (for rare cases like loaded data)
        /// Most Q8_1 tensors should be created as mutable activation buffers.
        Q8_1Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);

        /// Construct from Q8_1Block array (for testing and direct block construction)
        /// @param shape Tensor dimensions [rows, cols]
        /// @param blocks Pointer to Q8_1Block array (copied)
        /// @param num_blocks Number of blocks in the array
        /// @param device Device placement (default: CPU)
        Q8_1Tensor(const std::vector<size_t> &shape, const Q8_1Block *blocks, size_t num_blocks, DeviceId device = DeviceId::cpu());

        /// Copy constructor (deep copy of all data)
        Q8_1Tensor(const Q8_1Tensor &other);

        /// Construct mutable Q8_1 activation buffer - the primary use case
        /// Use mutable_q8_1_blocks() to write Q8_1 data, NOT mutable_data().
        /// @param shape Tensor dimensions [rows, cols]
        /// @param device Device placement (default: CPU)
        explicit Q8_1Tensor(const std::vector<size_t> &shape, DeviceId device = DeviceId::cpu());

        ~Q8_1Tensor() override;

        // TensorBase interface
        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q8_1; }

        DeviceId home_device() const override { return device_; }

        /**
         * @brief DEPRECATED: Do not use data() on Q8_1Tensor - throws std::runtime_error
         * @note This override intentionally throws to catch accidental FP32 dequantization.
         *       Use fp32_data() if you explicitly need FP32 values, or q8_1_blocks() for
         *       native Q8_1 block access in performance-critical code.
         * @see fp32_data(), q8_1_blocks()
         */
        const float *data() const override;

        /**
         * @brief Explicitly dequantize Q8_1 blocks to FP32 buffer
         * @return Pointer to cached FP32 buffer containing dequantized values
         * @note This method explicitly acknowledges the precision conversion.
         *       Results are cached until Q8_1 blocks are modified via mutable_q8_1_blocks().
         *       Prefer q8_1_blocks() for kernels that can operate on Q8_1 natively.
         * @see q8_1_blocks() for native Q8_1 access without dequantization
         */
        const float *fp32_data() const override;

        float *mutable_data() override;

        // ===== Diamond Inheritance Resolution =====
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        bool copyFrom(const TensorBase *src) override;

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Memory management - release raw data after GEMM packing
        void release_raw_data() override
        {
            if (!is_view_ && !raw_data_released_)
            {
                raw_data_.clear();
                raw_data_.shrink_to_fit();
                raw_data_released_ = true;
            }
        }
        bool is_raw_data_released() const override { return raw_data_released_; }

        void release_host_weight_data() override
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
            {
                decltype(dequant_cache_) tmp;
                std::swap(dequant_cache_, tmp);
            }
        }

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

        // IINT8Unpackable interface
        void unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const override
        {
            const Q8_1Block *block = static_cast<const Q8_1Block *>(get_raw_block_at(row_idx, k_block_offset));
            std::memcpy(output, block->qs, 32);
        }

        float get_block_scale(size_t row_idx, size_t k_block_offset) const override
        {
            const Q8_1Block *block = static_cast<const Q8_1Block *>(get_raw_block_at(row_idx, k_block_offset));
            return fp16_to_fp32(block->d);
        }

        size_t superblock_size() const override { return 256; }

        /// Efficient override: reads Q8_1 blocks directly via typed_data()
        /// without virtual dispatch per block.
        float requantizeRowToInt8(size_t row_idx, size_t K, int8_t *output) const override;

        void unpack_superblock_to_int8(
            size_t row_idx,
            size_t superblock_idx,
            int8_t *output,
            float *scales = nullptr,
            float *mins = nullptr) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q8_1Block *blocks = reinterpret_cast<const Q8_1Block *>(data_ptr);
            const size_t base_block = superblock_idx * 8;

            // Copy 8 consecutive blocks (8 × 32 = 256 bytes of int8 data)
            const Q8_1Block *src = &blocks[row_idx * blocks_per_row + base_block];
            for (int i = 0; i < 8; ++i)
            {
                std::memcpy(output + i * 32, src[i].qs, 32);
            }

            if (scales)
            {
                for (int i = 0; i < 8; ++i)
                {
                    scales[i] = fp16_to_fp32(src[i].d);
                }
            }
            if (mins)
            {
                for (int i = 0; i < 8; ++i)
                {
                    mins[i] = 0.0f; // Q8_1 is symmetric (s field is sum, not min)
                }
            }
        }

        // IActivationTensor interface (Q8_1 activations)
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;
        std::unique_ptr<ITensorAttention> createAttention() override;
        std::unique_ptr<ITensorEmbedding> createEmbedding() override;
        ActivationPack to_int8_activation_pack(int rows, int cols) const override;
        bool quantize_to_q8_1(void *q8_1_buffer, int m, int k) const override;

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

        bool applyRMSNorm(
            const float *gamma,
            int seq_len,
            int d_model,
            float eps = 1e-6f,
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

        /**
         * @brief Check if tensor is mutable (activation buffer with FP32 cache)
         * @return true if tensor was created as mutable activation buffer
         */
        bool is_mutable() const { return is_mutable_; }

        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        // ===== Q8_1 Block-Aligned K-Slicing (Phase 6.4) =====

        /**
         * @brief Check if a k-slice offset is block-aligned (multiple of 32)
         * @param k_start Starting column index for k-slice
         * @return true if k_start is aligned to Q8_1Block::BLOCK_SIZE (32) boundary
         */
        static constexpr bool is_k_aligned(size_t k_start)
        {
            return (k_start % Q8_1Block::BLOCK_SIZE) == 0;
        }

        /**
         * @brief Create a k-sliced view (column slice) at block boundaries
         *
         * This enables block-native k-slicing for tensor parallelism without
         * FP32 dequantization. Only works when k_start is aligned to 32-element
         * block boundaries.
         *
         * @param k_start Starting column index (must be multiple of 32)
         * @param k_size Number of columns in slice (must be multiple of 32, or extend to end)
         * @return New Q8_1Tensor that's a k-sliced view, or nullptr if alignment fails
         *
         * @note Unlike create_view() which slices rows, this slices columns (K dimension).
         *       The returned tensor has shape [rows, k_size] with blocks copied to
         *       a new contiguous buffer (views for k-slicing are not zero-copy since
         *       Q8_1 blocks are row-major and we need column-contiguous blocks).
         *
         * Usage pattern (tensor parallelism k-sliced GEMM):
         *   const int k_local = k / world_size;
         *   const int k_start = k_local * rank;
         *   if (Q8_1Tensor::is_k_aligned(k_start) && Q8_1Tensor::is_k_aligned(k_local)) {
         *       auto input_local = input_q8_1->slice_k_blocks(k_start, k_local);
         *       // GEMM with Q8_1 input directly - no FP32 dequantization!
         *   } else {
         *       // Fallback to FP32 path with fp32_data() dequantization
         *   }
         */
        std::shared_ptr<Q8_1Tensor> slice_k_blocks(size_t k_start, size_t k_size) const;

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

        // ===== Q8_1-Specific Native Accessors =====
        // These provide direct access to Q8_1Block storage for typed kernels.
        // Use these instead of data()/mutable_data() when working with Q8_1 activations.

        /**
         * @brief Get const pointer to Q8_1 block storage
         * @return Pointer to first Q8_1Block (row-major, blocks_per_row = ceil(cols/32))
         */
        const Q8_1Block *q8_1_blocks() const
        {
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            return reinterpret_cast<const Q8_1Block *>(data_ptr);
        }

        /**
         * @brief Get mutable pointer to Q8_1 block storage
         * @return Pointer to first Q8_1Block for in-place modification
         * @note Invalidates dequant_cache_ - call clear_dequant_cache() if needed
         */
        Q8_1Block *mutable_q8_1_blocks()
        {
            // Invalidate dequant cache since blocks are being modified
            dequant_cache_.clear();

            // For views, also invalidate parent's cache since we share underlying storage
            // and any modification through this view invalidates the parent's cached dequantized data
            if (is_view_ && parent_)
            {
                auto parent_q8_1 = std::dynamic_pointer_cast<Q8_1Tensor>(parent_);
                if (parent_q8_1)
                {
                    parent_q8_1->clear_dequant_cache();
                }
            }

            uint8_t *data_ptr = is_view_
                                    ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_)
                                    : raw_data_.data();
            return reinterpret_cast<Q8_1Block *>(data_ptr);
        }

        /**
         * @brief Get number of Q8_1 blocks per row
         * @return blocks_per_row = ceil(cols / 32)
         */
        size_t blocks_per_row() const
        {
            return (shape_[1] + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
        }

        /**
         * @brief Get total number of Q8_1 blocks
         * @return rows * blocks_per_row
         */
        size_t total_blocks() const
        {
            return shape_[0] * blocks_per_row();
        }

        /**
         * @brief Clear the FP32 dequantization cache
         *
         * Call this after modifying blocks via mutable_q8_1_blocks() if you
         * subsequently need to use data() (which returns dequantized FP32).
         */
        void clear_dequant_cache() { dequant_cache_.clear(); }

        /**
         * @brief Re-quantize the FP32 dequant cache back to Q8_1 blocks
         *
         * Call this after writing to mutable_data() to commit FP32 changes
         * back to the native Q8_1 block storage. This enables the pattern:
         *   1. kernel writes FP32 to mutable_data()
         *   2. quantize_from_cache() converts to Q8_1
         *   3. next kernel reads native Q8_1 via q8_1_blocks()
         *
         * @return true on success
         */
        bool quantize_from_cache();

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

        /**
         * @brief Quantize FP32 data into existing Q8_1 tensor storage
         * @param src_data Source FP32 data with at least rows()*cols() values
         * @return true on success
         */
        bool copyFrom_fp32(const float *src_data);

        /**
         * @brief Quantize first num_rows rows from FP32 into existing Q8_1 tensor storage
         * @param src_data Source FP32 data with at least num_rows*cols() values
         * @param num_rows Number of rows to quantize (must be <= rows())
         * @return true on success
         */
        bool copyFrom_fp32_rows(const float *src_data, size_t num_rows);

    protected:
        // ===== Lazy Transfer Accessors (Phase 3) =====
        void *raw_host_data_ptr() override
        {
            return is_view_ ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        const void *raw_host_data_ptr() const override
        {
            return is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        size_t byte_size() const override { return raw_data_.size(); }

    private:
        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)

        DeviceId device_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        bool raw_data_released_ = false;   // Track if raw data was released after GEMM pack
        bool is_mutable_ = false;          // True if constructed as mutable activation buffer
        mutable bool cache_dirty_ = false; // True if dequant_cache_ has uncommitted writes

        // Private view constructor
        Q8_1Tensor(const std::vector<size_t> &shape,
                   const uint8_t *parent_raw_data,
                   size_t byte_offset,
                   std::shared_ptr<TensorBase> parent);
    };

    // ===== Q16_1 Tensor (16-bit quantization with pre-computed sum) =====

    // Implementation: Q16_1Tensor.cpp
    /**
     * @brief Q16_1 quantized tensor (16-bit with pre-computed sum, high-precision residual format)
     *
     * Like Q8_1 but with 256× more precision (int16 vs int8). Designed for residual stream
     * where quantization error accumulation is critical. Uses same block structure and
     * integer-domain computation patterns as Q8_1.
     *
     * Block format: 32 elements per block (72 bytes)
     * - FP16 scale factor (d)
     * - INT32 pre-computed sum (sum_qs) - raw integer sum
     * - 32× INT16 quantized values (qs)
     *
     * Memory: 2.25 bytes/element (vs 1.125 for Q8_1, 4 for FP32)
     * Range: [-32767, 32767] per element (vs [-127, 127] for Q8_1)
     *
     * GEMM pattern (matching Q8_1):
     * - Q16_1 can be dequantized to FP32 for FP32 GEMM
     * - Q16_1 can be converted to Q8_1 for Q8_1-native GEMM (with some precision loss)
     * - Future: native Q16_1×Q8_1 GEMM kernels
     *
     * Use case: High-precision residual stream quantization where error accumulation
     * across 20+ transformer layers causes token prediction divergence with Q8_1.
     */
    class Q16_1Tensor : public TypedTensorBase<Q16_1Tensor, Q16_1Block>,
                        public TensorBase,
                        public IActivationTensor,
                        public ITensorGemmTileDataProvider
    {
    public:
        /// Native storage type (same as TypedTensorBase::value_type)
        using value_type = Q16_1Block;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::Q16_1; }

        // ===== CRTP Implementation for TypedTensorBase =====
        // NOTE: These delegate to the deprecated q16_1_blocks() methods.
        // The TypedTensorBase::typed_data() interface is inherently unsafe for Q16_1Tensor
        // because it always returns Q16_1Block* regardless of actual block size.
        // Use as_block_32/64/128() or dispatchQ16Block() for safe access.

        /// Called by TypedTensorBase::typed_data() - returns Q16_1Block*
        /// @deprecated typed_data() is unsafe for Q16_1Tensor; use as_block_32/64/128()
        const Q16_1Block *data_impl() const
        {
            // Delegate to deprecated accessor (which logs warning in debug builds)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            return q16_1_blocks();
#pragma GCC diagnostic pop
        }
        /// Called by TypedTensorBase::mutable_typed_data() - returns mutable Q16_1Block*
        /// @deprecated mutable_typed_data() is unsafe for Q16_1Tensor; use mutable_as_block_32/64/128()
        Q16_1Block *mutable_data_impl()
        {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            return mutable_q16_1_blocks();
#pragma GCC diagnostic pop
        }

        /// Convenience alias: blocks() returns typed_data()
        /// @deprecated Use as_block_32/64/128() or dispatchQ16Block() for safe access
        [[deprecated("Use as_block_32/64/128() or dispatchQ16Block() for safe block access")]]
        const Q16_1Block *blocks() const
        {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            return typed_data();
#pragma GCC diagnostic pop
        }
        /// @deprecated Use mutable_as_block_32/64/128() for safe access
        [[deprecated("Use mutable_as_block_32/64/128() for safe block access")]]
        Q16_1Block *mutable_blocks()
        {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            return mutable_typed_data();
#pragma GCC diagnostic pop
        }

        /// Construct from existing raw Q16_1 block data
        Q16_1Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);

        /// Construct from Q16_1Block array (for testing and direct block construction)
        /// @param device Device placement (default: CPU)
        Q16_1Tensor(const std::vector<size_t> &shape, const Q16_1Block *blocks, size_t num_blocks, DeviceId device = DeviceId::cpu());

        /// Copy constructor (deep copy of all data)
        Q16_1Tensor(const Q16_1Tensor &other);

        /// Construct mutable Q16_1 activation buffer - the primary use case
        /// @param device Device placement (default: CPU)
        explicit Q16_1Tensor(const std::vector<size_t> &shape, DeviceId device = DeviceId::cpu());

        /// Construct mutable Q16_1 activation buffer with custom block size
        /// @param shape Tensor shape [rows, cols]
        /// @param block_size Block size for quantization (32, 64, 128, or 192)
        /// @param device Device placement (default: CPU)
        Q16_1Tensor(const std::vector<size_t> &shape, Q16BlockSize block_size, DeviceId device = DeviceId::cpu());

        ~Q16_1Tensor() override;

        // TensorBase interface
        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q16_1; }

        DeviceId home_device() const override { return device_; }

        const float *data() const override;
        const float *fp32_data() const override;
        float *mutable_data() override;

        // ===== Diamond Inheritance Resolution =====
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        bool copyFrom(const TensorBase *src) override;

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Memory management
        void release_raw_data() override;
        bool is_raw_data_released() const override { return raw_data_released_; }

        void release_host_weight_data() override
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
            {
                decltype(dequant_cache_) tmp;
                std::swap(dequant_cache_, tmp);
            }
        }

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

        // IActivationTensor interface
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;
        std::unique_ptr<ITensorAttention> createAttention() override;
        std::unique_ptr<ITensorEmbedding> createEmbedding() override;
        ActivationPack to_int8_activation_pack(int rows, int cols) const override;
        bool quantize_to_q8_1(void *q8_1_buffer, int m, int k) const override;

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

        bool applyRMSNorm(
            const float *gamma,
            int seq_len,
            int d_model,
            float eps = 1e-6f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        bool from_int32_with_scales(
            const int32_t *accum,
            int rows,
            int cols,
            const float *row_scales,
            const float *col_scales,
            const float *bias = nullptr) override;

        // View support
        bool is_view() const override { return is_view_; }
        bool is_mutable() const { return is_mutable_; }

        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

        // ITensorGemmTileDataProvider interface
        void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override;

        const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override;

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return static_cast<size_t>(block_size_); }

        /// Get the block size enum value
        Q16BlockSize q16_block_size() const { return block_size_; }

        /**
         * @brief Get dtype name including block size suffix for stage dumps
         *
         * Returns "Q16_1_32", "Q16_1_64", or "Q16_1_128" depending on block size.
         * This enables stage dump infrastructure to correctly compute byte sizes
         * for Q16_1 tensors with variable block sizes.
         *
         * @return Static string like "Q16_1_64"
         */
        const char *dtype_name_with_block_size() const
        {
            switch (block_size_)
            {
            case Q16BlockSize::BLOCK_32:
                return "Q16_1_32";
            case Q16BlockSize::BLOCK_64:
                return "Q16_1_64";
            case Q16BlockSize::BLOCK_128:
                return "Q16_1_128";
            default:
                return "Q16_1_32"; // Fallback
            }
        }

        // =================================================================
        // Safe Block Accessors (Phase 2 API)
        // Return nullptr if block size doesn't match - use for safe runtime dispatch
        // =================================================================

        /// Get Q16_1Block* if this tensor uses BLOCK_32, else nullptr
        const Q16_1Block *as_block_32() const
        {
            if (block_size_ != Q16BlockSize::BLOCK_32)
                return nullptr;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            return reinterpret_cast<const Q16_1Block *>(data_ptr);
        }
        Q16_1Block *mutable_as_block_32()
        {
            if (block_size_ != Q16BlockSize::BLOCK_32)
                return nullptr;
            clear_dequant_cache_and_parent();
            uint8_t *data_ptr = is_view_
                                    ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_)
                                    : raw_data_.data();
            return reinterpret_cast<Q16_1Block *>(data_ptr);
        }

        /// Get Q16_1Block_64* if this tensor uses BLOCK_64, else nullptr
        const Q16_1Block_64 *as_block_64() const
        {
            if (block_size_ != Q16BlockSize::BLOCK_64)
                return nullptr;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            return reinterpret_cast<const Q16_1Block_64 *>(data_ptr);
        }
        Q16_1Block_64 *mutable_as_block_64()
        {
            if (block_size_ != Q16BlockSize::BLOCK_64)
                return nullptr;
            clear_dequant_cache_and_parent();
            uint8_t *data_ptr = is_view_
                                    ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_)
                                    : raw_data_.data();
            return reinterpret_cast<Q16_1Block_64 *>(data_ptr);
        }

        /// Get Q16_1Block_128* if this tensor uses BLOCK_128, else nullptr
        const Q16_1Block_128 *as_block_128() const
        {
            if (block_size_ != Q16BlockSize::BLOCK_128)
                return nullptr;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            return reinterpret_cast<const Q16_1Block_128 *>(data_ptr);
        }
        Q16_1Block_128 *mutable_as_block_128()
        {
            if (block_size_ != Q16BlockSize::BLOCK_128)
                return nullptr;
            clear_dequant_cache_and_parent();
            uint8_t *data_ptr = is_view_
                                    ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_)
                                    : raw_data_.data();
            return reinterpret_cast<Q16_1Block_128 *>(data_ptr);
        }

        // =================================================================
        // Generic Element Access (slower but always safe)
        // =================================================================

        /// Get dequantized element at (row, col) - handles any block size
        float dequant_element(size_t row, size_t col) const
        {
            const size_t block_elems = q16_block_size_elements(block_size_);
            const size_t block_bytes = q16_block_size_bytes(block_size_);
            const size_t bpr = blocks_per_row();

            const size_t b = col / block_elems;
            const size_t i = col % block_elems;
            const size_t block_idx = row * bpr + b;

            const uint8_t *base_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const uint8_t *block_ptr = base_ptr + block_idx * block_bytes;
            float d;
            std::memcpy(&d, block_ptr, sizeof(float)); // Scale at offset 0
            const int16_t *qs = reinterpret_cast<const int16_t *>(block_ptr + sizeof(float) + sizeof(int32_t));

            return d * static_cast<float>(qs[i]);
        }

        /// Get quantized int16 element at (row, col)
        int16_t quantized_element(size_t row, size_t col) const
        {
            const size_t block_elems = q16_block_size_elements(block_size_);
            const size_t block_bytes = q16_block_size_bytes(block_size_);
            const size_t bpr = blocks_per_row();

            const size_t b = col / block_elems;
            const size_t i = col % block_elems;
            const size_t block_idx = row * bpr + b;

            const uint8_t *base_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const uint8_t *block_ptr = base_ptr + block_idx * block_bytes;
            const int16_t *qs = reinterpret_cast<const int16_t *>(block_ptr + sizeof(float) + sizeof(int32_t));

            return qs[i];
        }

        /// Get block scale at (row, block_in_row)
        float block_scale(size_t row, size_t block_in_row) const
        {
            const size_t block_bytes = q16_block_size_bytes(block_size_);
            const size_t bpr = blocks_per_row();
            const size_t block_idx = row * bpr + block_in_row;

            const uint8_t *base_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const uint8_t *block_ptr = base_ptr + block_idx * block_bytes;
            float d;
            std::memcpy(&d, block_ptr, sizeof(float));
            return d;
        }

        // ===== Q16_1-Specific Native Accessors (Legacy - use safe accessors above) =====
        // WARNING: These methods assume BLOCK_32 and return Q16_1Block* unconditionally.
        // For tensors with BLOCK_64 or BLOCK_128, using these methods leads to memory
        // corruption. Use as_block_32/64/128() or dispatchQ16Block() instead.

        /// @deprecated Use as_block_32() or dispatchQ16Block() for safe access
        [[deprecated("Use as_block_32/64/128() or dispatchQ16Block() for safe block access")]]
        const Q16_1Block *q16_1_blocks() const
        {
#if LLAMINAR_ASSERTIONS_ACTIVE
            if (block_size_ != Q16BlockSize::BLOCK_32)
            {
                LOG_ERROR("[Q16_1Tensor] q16_1_blocks() called on tensor with block_size="
                          << static_cast<int>(block_size_) << " (expected 32)");
                LOG_ERROR("[Q16_1Tensor] Use as_block_64() or as_block_128() for this tensor");
            }
#endif
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            return reinterpret_cast<const Q16_1Block *>(data_ptr);
        }

        /// @deprecated Use mutable_as_block_32() or dispatchQ16BlockMutable() for safe access
        [[deprecated("Use mutable_as_block_32/64/128() or dispatchQ16BlockMutable() for safe block access")]]
        Q16_1Block *mutable_q16_1_blocks()
        {
#if LLAMINAR_ASSERTIONS_ACTIVE
            if (block_size_ != Q16BlockSize::BLOCK_32)
            {
                LOG_ERROR("[Q16_1Tensor] mutable_q16_1_blocks() called on tensor with block_size="
                          << static_cast<int>(block_size_) << " (expected 32)");
                LOG_ERROR("[Q16_1Tensor] Use mutable_as_block_64() or mutable_as_block_128() for this tensor");
            }
#endif
            clear_dequant_cache_and_parent();
            uint8_t *data_ptr = is_view_
                                    ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_)
                                    : raw_data_.data();
            return reinterpret_cast<Q16_1Block *>(data_ptr);
        }

        size_t blocks_per_row() const
        {
            const size_t bs = static_cast<size_t>(block_size_);
            return (shape_[1] + bs - 1) / bs;
        }

        size_t total_blocks() const
        {
            return shape_[0] * blocks_per_row();
        }

        void clear_dequant_cache() { dequant_cache_.clear(); }

        bool quantize_from_cache();

        // Convert Q16_1 to Q8_1 (with precision reduction)
        std::shared_ptr<Q8_1Tensor> to_q8_1() const;

        // Public decode methods
        static void decodeBlock(const Q16_1Block &block, float *output);
        static void decodeBlockScalar(const Q16_1Block &block, float *output);

#if defined(__AVX512F__)
        static void decodeBlockAVX512(const Q16_1Block &block, float *output);
#endif

#if defined(__AVX2__)
        static void decodeBlockAVX2(const Q16_1Block &block, float *output);
#endif

        /**
         * @brief Quantize FP32 data to Q16_1 format with pre-computed sum
         * @param src Source FP32 data
         * @param shape Tensor shape [rows, cols]
         * @return Q16_1Tensor with pre-computed sums
         */
        static std::shared_ptr<Q16_1Tensor> quantize_from_fp32(
            const float *src,
            const std::vector<size_t> &shape);

    protected:
        void *raw_host_data_ptr() override
        {
            return is_view_ ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        const void *raw_host_data_ptr() const override
        {
            return is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        size_t byte_size() const override { return raw_data_.size(); }

    private:
        std::vector<size_t> shape_;

        bool is_view_;
        std::vector<uint8_t> raw_data_;
        const uint8_t *raw_data_ptr_;
        size_t view_byte_offset_;
        std::shared_ptr<TensorBase> parent_;

        DeviceId device_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        bool raw_data_released_ = false;
        bool is_mutable_ = false;
        mutable bool cache_dirty_ = false;

        /// Block size for Q16_1 quantization (32, 64, 128, or 192 elements per block)
        /// Default is BLOCK_32 for backward compatibility with existing code
        Q16BlockSize block_size_ = Q16BlockSize::BLOCK_32;

        /// Helper to clear dequant cache (and parent's cache if this is a view)
        void clear_dequant_cache_and_parent()
        {
            dequant_cache_.clear();
            if (is_view_ && parent_)
            {
                auto parent_q16_1 = std::dynamic_pointer_cast<Q16_1Tensor>(parent_);
                if (parent_q16_1)
                {
                    parent_q16_1->clear_dequant_cache();
                }
            }
        }

    public:
        // Copy from FP32 data (quantizes to Q16_1 blocks)
        bool copyFrom_fp32(const float *src_data);

        // Copy from FP32 data for only the first num_rows rows
        // Used for embedding output where we only fill part of the buffer
        bool copyFrom_fp32_rows(const float *src_data, size_t num_rows);

        /**
         * @brief Copy from FP32 data using FIXED-SCALE quantization with VNNI-safe clipping
         *
         * CRITICAL: This method MUST be used for Q/K/V tensors in integer attention.
         * Unlike copyFrom_fp32() which uses adaptive per-block scaling, this uses a
         * fixed scale and clips INT16 values to prevent VNNI INT32 overflow.
         *
         * Formula: int16 = clip(round(fp32 * 32767 / kv_cache_scale), ±MAX_SAFE_INT16)
         *
         * See: kernels/cpu/attention/q16_1/VNNISafetyConstants.h for limits
         * See: docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md "VNNI OVERFLOW PREVENTION CONTRACT"
         *
         * @param src_data Source FP32 data
         * @param kv_cache_scale The fixed scale factor (e.g., 8.0 for ±8.0 FP32 range)
         * @param head_dim The attention head dimension (for VNNI safety limits)
         * @return true if successful
         */
        bool copyFrom_fp32_fixed_scale(const float *src_data, float kv_cache_scale, int head_dim);

        /**
         * @brief Copy from FP32 data for only the first num_rows rows with fixed-scale quantization
         *
         * CRITICAL: This method MUST be used for Q/K/V tensors in integer attention.
         *
         * @param src_data Source FP32 data
         * @param num_rows Number of rows to copy
         * @param kv_cache_scale The fixed scale factor (e.g., 8.0 for ±8.0 FP32 range)
         * @param head_dim The attention head dimension (for VNNI safety limits)
         * @return true if successful
         */
        bool copyFrom_fp32_rows_fixed_scale(const float *src_data, size_t num_rows, float kv_cache_scale, int head_dim);

    private:
        // Private view constructor
        Q16_1Tensor(const std::vector<size_t> &shape,
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
    class Q4_0Tensor : public TypedTensorBase<Q4_0Tensor, Q4_0Block>, public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable, public IINT8Unpackable
    {
    public:
        /// Native storage type for CRTP-style type-safe access
        using value_type = Q4_0Block;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::Q4_0; }

        // TypedTensorBase CRTP implementation
        const Q4_0Block *data_impl() const { return reinterpret_cast<const Q4_0Block *>(raw_host_data_ptr()); }
        Q4_0Block *mutable_data_impl() { throw std::runtime_error("Q4_0Tensor: weight tensors are read-only"); }
        const Q4_0Block *blocks() const { return typed_data(); }

        Q4_0Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        /// Zero-copy constructor for mmap-backed data (no memcpy)
        Q4_0Tensor(const std::vector<size_t> &shape, const uint8_t *mmap_data,
                   size_t byte_size, std::shared_ptr<void> mmap_lifetime_owner);
        ~Q4_0Tensor() override;

        // TensorBase interface
        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q4_0; }

        DeviceId home_device() const override { return device_; }

        const float *data() const override;
        float *mutable_data() override;

        // Diamond inheritance resolution (ITensor implemented by both TypedTensorBase and TensorBase)
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Memory management - release raw data after GEMM packing
        void release_raw_data() override
        {
            if (!is_view_ && !raw_data_released_)
            {
                raw_data_.clear();
                raw_data_.shrink_to_fit();
                raw_data_released_ = true;
            }
        }
        bool is_raw_data_released() const override { return raw_data_released_; }

        void release_host_weight_data() override
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
            {
                decltype(dequant_cache_) tmp;
                std::swap(dequant_cache_, tmp);
            }
            mmap_owner_.reset();
        }

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

        // IQ8_0Decodable interface
        void decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const override;

        // IINT8Unpackable interface - native Q4_0 → INT8 unpacking
        void unpack_block_to_int8(
            size_t row_idx,
            size_t k_block_offset,
            int8_t *output) const override;

        float get_block_scale(
            size_t row_idx,
            size_t k_block_offset) const override;

        size_t superblock_size() const override { return 256; }
        const NativeVnniFormatInfo *vnniFormatInfo() const override
        {
            static constexpr NativeVnniFormatInfo info{0, 16, false, false, false, 8.0f};
            return &info;
        }
        void packVnniBlock(const VnniPackContext &ctx, int n, int b) const override;

        void unpack_superblock_to_int8(
            size_t row_idx,
            size_t superblock_idx,
            int8_t *output,
            float *scales = nullptr,
            float *mins = nullptr) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q4_0Block::BLOCK_SIZE - 1) / Q4_0Block::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q4_0Block *blocks = reinterpret_cast<const Q4_0Block *>(data_ptr);
            const size_t base_block = superblock_idx * 8;
            const Q4_0Block *src = &blocks[row_idx * blocks_per_row + base_block];

            // Unpack 8 consecutive blocks (8 × 32 = 256 elements)
            for (int i = 0; i < 8; ++i)
            {
                simd::unpack_q4_0_to_int8(src[i], output + i * 32);
            }

            if (scales)
            {
                for (int i = 0; i < 8; ++i)
                {
                    scales[i] = fp16_to_fp32(src[i].d);
                }
            }
            if (mins)
            {
                for (int i = 0; i < 8; ++i)
                {
                    mins[i] = 0.0f; // Q4_0 is symmetric
                }
            }
        }

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

    protected:
        // ===== Lazy Transfer Accessors (Phase 3) =====
        void *raw_host_data_ptr() override
        {
            return is_view_ ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        const void *raw_host_data_ptr() const override
        {
            return is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        size_t byte_size() const override
        {
            // Calculate from shape, not raw_data_.size() (which may be cleared after GEMM pack)
            // Q4_0: 32 elements per block, 18 bytes per block
            size_t blocks_per_row = (shape_[1] + Q4_0Block::BLOCK_SIZE - 1) / Q4_0Block::BLOCK_SIZE;
            return shape_[0] * blocks_per_row * sizeof(Q4_0Block);
        }

    private:
        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)
        std::shared_ptr<void> mmap_owner_;   // Keeps mmap region alive for zero-copy tensors
        size_t data_byte_size_ = 0;          // Byte size for mmap-backed tensors (raw_data_ is empty)

        DeviceId device_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        bool raw_data_released_ = false; // Track if raw data was released after GEMM pack

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
    class Q4_1Tensor : public TypedTensorBase<Q4_1Tensor, Q4_1Block>, public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable, public IINT8Unpackable
    {
    public:
        /// Native storage type for CRTP-style type-safe access
        using value_type = Q4_1Block;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::Q4_1; }

        // TypedTensorBase CRTP implementation
        const Q4_1Block *data_impl() const { return reinterpret_cast<const Q4_1Block *>(raw_host_data_ptr()); }
        Q4_1Block *mutable_data_impl() { throw std::runtime_error("Q4_1Tensor: weight tensors are read-only"); }
        const Q4_1Block *blocks() const { return typed_data(); }

        Q4_1Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        /// Zero-copy constructor for mmap-backed data (no memcpy)
        Q4_1Tensor(const std::vector<size_t> &shape, const uint8_t *mmap_data,
                   size_t byte_size, std::shared_ptr<void> mmap_lifetime_owner);
        ~Q4_1Tensor() override;

        // TensorBase interface
        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q4_1; }

        DeviceId home_device() const override { return device_; }

        const float *data() const override;
        float *mutable_data() override;

        // Diamond inheritance resolution (ITensor implemented by both TypedTensorBase and TensorBase)
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Memory management - release raw data after GEMM packing
        void release_raw_data() override
        {
            if (!is_view_ && !raw_data_released_)
            {
                raw_data_.clear();
                raw_data_.shrink_to_fit();
                raw_data_released_ = true;
            }
        }
        bool is_raw_data_released() const override { return raw_data_released_; }

        void release_host_weight_data() override
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
            {
                decltype(dequant_cache_) tmp;
                std::swap(dequant_cache_, tmp);
            }
            mmap_owner_.reset();
        }

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

        // IINT8Unpackable interface - native Q4_1 → INT8 unpacking
        void unpack_block_to_int8(
            size_t row_idx,
            size_t k_block_offset,
            int8_t *output) const override;

        float get_block_scale(
            size_t row_idx,
            size_t k_block_offset) const override;

        float get_block_min(
            size_t row_idx,
            size_t k_block_offset) const override;

        size_t superblock_size() const override { return 256; }
        const NativeVnniFormatInfo *vnniFormatInfo() const override
        {
            static constexpr NativeVnniFormatInfo info{5, 16, true, false, false, 15.0f};
            return &info;
        }
        void packVnniBlock(const VnniPackContext &ctx, int n, int b) const override;

        void unpack_superblock_to_int8(
            size_t row_idx,
            size_t superblock_idx,
            int8_t *output,
            float *scales = nullptr,
            float *mins = nullptr) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q4_1Block::BLOCK_SIZE - 1) / Q4_1Block::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q4_1Block *blocks = reinterpret_cast<const Q4_1Block *>(data_ptr);
            const size_t base_block = superblock_idx * 8;
            const Q4_1Block *src = &blocks[row_idx * blocks_per_row + base_block];

            // Unpack 8 consecutive blocks (8 × 32 = 256 elements)
            for (int i = 0; i < 8; ++i)
            {
                simd::unpack_q4_1_to_int8(src[i], output + i * 32);
            }

            if (scales)
            {
                for (int i = 0; i < 8; ++i)
                {
                    scales[i] = fp16_to_fp32(src[i].d);
                }
            }
            if (mins)
            {
                for (int i = 0; i < 8; ++i)
                {
                    mins[i] = fp16_to_fp32(src[i].m);
                }
            }
        }

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

    protected:
        // ===== Lazy Transfer Accessors (Phase 3) =====
        void *raw_host_data_ptr() override
        {
            return is_view_ ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        const void *raw_host_data_ptr() const override
        {
            return is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        size_t byte_size() const override { return data_byte_size_ > 0 ? data_byte_size_ : raw_data_.size(); }

    private:
        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)
        std::shared_ptr<void> mmap_owner_;   // Keeps mmap region alive for zero-copy tensors
        size_t data_byte_size_ = 0;          // Byte size for mmap-backed tensors (raw_data_ is empty)

        DeviceId device_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        bool raw_data_released_ = false; // Track if raw data was released after GEMM pack

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
    class Q5_0Tensor : public TypedTensorBase<Q5_0Tensor, Q5_0Block>, public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable, public IINT8Unpackable
    {
    public:
        /// Native storage type for CRTP-style type-safe access
        using value_type = Q5_0Block;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::Q5_0; }

        // TypedTensorBase CRTP implementation
        const Q5_0Block *data_impl() const { return reinterpret_cast<const Q5_0Block *>(raw_host_data_ptr()); }
        Q5_0Block *mutable_data_impl() { throw std::runtime_error("Q5_0Tensor: weight tensors are read-only"); }
        const Q5_0Block *blocks() const { return typed_data(); }

        Q5_0Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        /// Zero-copy constructor for mmap-backed data (no memcpy)
        Q5_0Tensor(const std::vector<size_t> &shape, const uint8_t *mmap_data,
                   size_t byte_size, std::shared_ptr<void> mmap_lifetime_owner);
        ~Q5_0Tensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q5_0; }

        DeviceId home_device() const override { return device_; }

        const float *data() const override;
        float *mutable_data() override;

        // Diamond inheritance resolution (ITensor implemented by both TypedTensorBase and TensorBase)
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Memory management - release raw data after GEMM packing
        void release_raw_data() override
        {
            if (!is_view_ && !raw_data_released_)
            {
                raw_data_.clear();
                raw_data_.shrink_to_fit();
                raw_data_released_ = true;
            }
        }
        bool is_raw_data_released() const override { return raw_data_released_; }

        void release_host_weight_data() override
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
            {
                decltype(dequant_cache_) tmp;
                std::swap(dequant_cache_, tmp);
            }
            mmap_owner_.reset();
        }

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

        // IINT8Unpackable interface
        void unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const override;
        float get_block_scale(size_t row_idx, size_t k_block_offset) const override;
        float get_block_min(size_t row_idx, size_t k_block_offset) const override { return 0.0f; } // Q5_0 is symmetric

        size_t superblock_size() const override { return 256; }
        const NativeVnniFormatInfo *vnniFormatInfo() const override
        {
            static constexpr NativeVnniFormatInfo info{6, 20, false, false, false, 16.0f};
            return &info;
        }
        void packVnniBlock(const VnniPackContext &ctx, int n, int b) const override;

        void unpack_superblock_to_int8(
            size_t row_idx,
            size_t superblock_idx,
            int8_t *output,
            float *scales = nullptr,
            float *mins = nullptr) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q5_0Block::BLOCK_SIZE - 1) / Q5_0Block::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q5_0Block *blocks = reinterpret_cast<const Q5_0Block *>(data_ptr);
            const size_t base_block = superblock_idx * 8;
            const Q5_0Block *src = &blocks[row_idx * blocks_per_row + base_block];

            // Unpack 8 consecutive blocks (8 × 32 = 256 elements)
            for (int i = 0; i < 8; ++i)
            {
                simd::unpack_q5_0_to_int8(src[i], output + i * 32);
            }

            if (scales)
            {
                for (int i = 0; i < 8; ++i)
                {
                    scales[i] = fp16_to_fp32(src[i].d);
                }
            }
            if (mins)
            {
                for (int i = 0; i < 8; ++i)
                {
                    mins[i] = 0.0f; // Q5_0 is symmetric
                }
            }
        }

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

    protected:
        // ===== Lazy Transfer Accessors (Phase 3) =====
        void *raw_host_data_ptr() override
        {
            return is_view_ ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        const void *raw_host_data_ptr() const override
        {
            return is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        size_t byte_size() const override { return data_byte_size_ > 0 ? data_byte_size_ : raw_data_.size(); }

    private:
        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)
        std::shared_ptr<void> mmap_owner_;   // Keeps mmap region alive for zero-copy tensors
        size_t data_byte_size_ = 0;          // Byte size for mmap-backed tensors (raw_data_ is empty)

        DeviceId device_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        bool raw_data_released_ = false; // Track if raw data was released after GEMM pack

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
    class Q5_1Tensor : public TypedTensorBase<Q5_1Tensor, Q5_1Block>, public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable, public IINT8Unpackable
    {
    public:
        /// Native storage type for CRTP-style type-safe access
        using value_type = Q5_1Block;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::Q5_1; }

        // TypedTensorBase CRTP implementation
        const Q5_1Block *data_impl() const { return reinterpret_cast<const Q5_1Block *>(raw_host_data_ptr()); }
        Q5_1Block *mutable_data_impl() { throw std::runtime_error("Q5_1Tensor: weight tensors are read-only"); }
        const Q5_1Block *blocks() const { return typed_data(); }

        Q5_1Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        /// Zero-copy constructor for mmap-backed data (no memcpy)
        Q5_1Tensor(const std::vector<size_t> &shape, const uint8_t *mmap_data,
                   size_t byte_size, std::shared_ptr<void> mmap_lifetime_owner);
        ~Q5_1Tensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q5_1; }

        DeviceId home_device() const override { return device_; }

        const float *data() const override;
        float *mutable_data() override;

        // Diamond inheritance resolution (ITensor implemented by both TypedTensorBase and TensorBase)
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Memory management - release raw data after GEMM packing
        void release_raw_data() override
        {
            if (!is_view_ && !raw_data_released_)
            {
                raw_data_.clear();
                raw_data_.shrink_to_fit();
                raw_data_released_ = true;
            }
        }
        bool is_raw_data_released() const override { return raw_data_released_; }

        void release_host_weight_data() override
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
            {
                decltype(dequant_cache_) tmp;
                std::swap(dequant_cache_, tmp);
            }
            mmap_owner_.reset();
        }

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

        // IINT8Unpackable interface
        void unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const override;
        float get_block_scale(size_t row_idx, size_t k_block_offset) const override;
        float get_block_min(size_t row_idx, size_t k_block_offset) const override;

        size_t superblock_size() const override { return 256; }
        const NativeVnniFormatInfo *vnniFormatInfo() const override
        {
            static constexpr NativeVnniFormatInfo info{7, 20, true, false, false, 31.0f};
            return &info;
        }
        void packVnniBlock(const VnniPackContext &ctx, int n, int b) const override;

        void unpack_superblock_to_int8(
            size_t row_idx,
            size_t superblock_idx,
            int8_t *output,
            float *scales = nullptr,
            float *mins = nullptr) const override
        {
            const size_t blocks_per_row = (shape_[1] + Q5_1Block::BLOCK_SIZE - 1) / Q5_1Block::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q5_1Block *blocks = reinterpret_cast<const Q5_1Block *>(data_ptr);
            const size_t base_block = superblock_idx * 8;
            const Q5_1Block *src = &blocks[row_idx * blocks_per_row + base_block];

            // Unpack 8 consecutive blocks (8 × 32 = 256 elements)
            for (int i = 0; i < 8; ++i)
            {
                simd::unpack_q5_1_to_int8(src[i], output + i * 32);
            }

            if (scales)
            {
                for (int i = 0; i < 8; ++i)
                {
                    scales[i] = fp16_to_fp32(src[i].d);
                }
            }
            if (mins)
            {
                for (int i = 0; i < 8; ++i)
                {
                    mins[i] = fp16_to_fp32(src[i].m);
                }
            }
        }

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

    protected:
        // ===== Lazy Transfer Accessors (Phase 3) =====
        void *raw_host_data_ptr() override
        {
            return is_view_ ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        const void *raw_host_data_ptr() const override
        {
            return is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        size_t byte_size() const override { return data_byte_size_ > 0 ? data_byte_size_ : raw_data_.size(); }

    private:
        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)
        std::shared_ptr<void> mmap_owner_;   // Keeps mmap region alive for zero-copy tensors
        size_t data_byte_size_ = 0;          // Byte size for mmap-backed tensors (raw_data_ is empty)

        DeviceId device_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        bool raw_data_released_ = false; // Track if raw data was released after GEMM pack

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
    class Q6_KTensor : public TypedTensorBase<Q6_KTensor, Q6_KBlock>, public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable, public IINT8Unpackable
    {
    public:
        /// Native storage type for CRTP-style type-safe access
        using value_type = Q6_KBlock;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::Q6_K; }

        // TypedTensorBase CRTP implementation
        const Q6_KBlock *data_impl() const { return reinterpret_cast<const Q6_KBlock *>(raw_host_data_ptr()); }
        Q6_KBlock *mutable_data_impl() { throw std::runtime_error("Q6_KTensor: weight tensors are read-only"); }
        const Q6_KBlock *blocks() const { return typed_data(); }

        Q6_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        /// Zero-copy constructor for mmap-backed data (no memcpy)
        Q6_KTensor(const std::vector<size_t> &shape, const uint8_t *mmap_data,
                   size_t byte_size, std::shared_ptr<void> mmap_lifetime_owner);
        ~Q6_KTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q6_K; }

        DeviceId home_device() const override { return device_; }

        const float *data() const override;
        float *mutable_data() override;

        // Diamond inheritance resolution (ITensor implemented by both TypedTensorBase and TensorBase)
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Memory management - release raw data after GEMM packing
        void release_raw_data() override
        {
            if (!is_view_ && !raw_data_released_)
            {
                raw_data_.clear();
                raw_data_.shrink_to_fit();
                raw_data_released_ = true;
            }
        }
        bool is_raw_data_released() const override { return raw_data_released_; }

        void release_host_weight_data() override
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
            {
                decltype(dequant_cache_) tmp;
                std::swap(dequant_cache_, tmp);
            }
            mmap_owner_.reset();
        }

        // IINT8Unpackable interface
        void unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const override;
        float get_block_scale(size_t row_idx, size_t k_block_offset) const override;
        float get_block_min(size_t row_idx, size_t k_block_offset) const override;
        size_t superblock_size() const override { return 256; }
        const NativeVnniFormatInfo *vnniFormatInfo() const override
        {
            static constexpr NativeVnniFormatInfo info{8, 24, true, true, false, 32.0f};
            return &info;
        }
        void packVnniBlock(const VnniPackContext &ctx, int n, int b) const override;
        void unpack_superblock_to_int8(size_t row_idx, size_t superblock_idx, int8_t *output, float *scales = nullptr, float *mins = nullptr) const override;

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

    protected:
        // ===== Lazy Transfer Accessors (Phase 3) =====
        void *raw_host_data_ptr() override
        {
            return is_view_ ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        const void *raw_host_data_ptr() const override
        {
            return is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        size_t byte_size() const override { return data_byte_size_ > 0 ? data_byte_size_ : raw_data_.size(); }

    private:
        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)
        std::shared_ptr<void> mmap_owner_;   // Keeps mmap region alive for zero-copy tensors
        size_t data_byte_size_ = 0;          // Byte size for mmap-backed tensors (raw_data_ is empty)

        DeviceId device_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        bool raw_data_released_ = false; // Track if raw data was released after GEMM pack

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
    class Q2_KTensor : public TypedTensorBase<Q2_KTensor, Q2_KBlock>, public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable, public IINT8Unpackable
    {
    public:
        /// Native storage type for CRTP-style type-safe access
        using value_type = Q2_KBlock;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::Q2_K; }

        // TypedTensorBase CRTP implementation
        const Q2_KBlock *data_impl() const { return reinterpret_cast<const Q2_KBlock *>(raw_host_data_ptr()); }
        Q2_KBlock *mutable_data_impl() { throw std::runtime_error("Q2_KTensor: weight tensors are read-only"); }
        const Q2_KBlock *blocks() const { return typed_data(); }

        Q2_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        /// Zero-copy constructor for mmap-backed data (no memcpy)
        Q2_KTensor(const std::vector<size_t> &shape, const uint8_t *mmap_data,
                   size_t byte_size, std::shared_ptr<void> mmap_lifetime_owner);
        ~Q2_KTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q2_K; }

        DeviceId home_device() const override { return device_; }

        const float *data() const override;
        float *mutable_data() override;

        // Diamond inheritance resolution (ITensor implemented by both TypedTensorBase and TensorBase)
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Memory management - release raw data after GEMM packing
        void release_raw_data() override
        {
            if (!is_view_ && !raw_data_released_)
            {
                raw_data_.clear();
                raw_data_.shrink_to_fit();
                raw_data_released_ = true;
            }
        }
        bool is_raw_data_released() const override { return raw_data_released_; }

        void release_host_weight_data() override
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
            {
                decltype(dequant_cache_) tmp;
                std::swap(dequant_cache_, tmp);
            }
            mmap_owner_.reset();
        }

        // IINT8Unpackable interface
        void unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const override;
        float get_block_scale(size_t row_idx, size_t k_block_offset) const override;
        float get_block_min(size_t row_idx, size_t k_block_offset) const override;
        size_t superblock_size() const override { return 256; }
        const NativeVnniFormatInfo *vnniFormatInfo() const override
        {
            static constexpr NativeVnniFormatInfo info{10, 8, true, true, true, 3.0f};
            return &info;
        }
        void packVnniBlock(const VnniPackContext &ctx, int n, int b) const override;
        void unpack_superblock_to_int8(size_t row_idx, size_t superblock_idx, int8_t *output, float *scales = nullptr, float *mins = nullptr) const override;

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

    protected:
        // ===== Lazy Transfer Accessors (Phase 3) =====
        void *raw_host_data_ptr() override
        {
            return is_view_ ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        const void *raw_host_data_ptr() const override
        {
            return is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        size_t byte_size() const override { return data_byte_size_ > 0 ? data_byte_size_ : raw_data_.size(); }

    private:
        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)
        std::shared_ptr<void> mmap_owner_;   // Keeps mmap region alive for zero-copy tensors
        size_t data_byte_size_ = 0;          // Byte size for mmap-backed tensors (raw_data_ is empty)

        DeviceId device_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        bool raw_data_released_ = false; // Track if raw data was released after GEMM pack

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
    class Q5_KTensor : public TypedTensorBase<Q5_KTensor, Q5_KBlock>, public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable, public IINT8Unpackable
    {
    public:
        /// Native storage type for CRTP-style type-safe access
        using value_type = Q5_KBlock;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::Q5_K; }

        // TypedTensorBase CRTP implementation
        const Q5_KBlock *data_impl() const { return reinterpret_cast<const Q5_KBlock *>(raw_host_data_ptr()); }
        Q5_KBlock *mutable_data_impl() { throw std::runtime_error("Q5_KTensor: weight tensors are read-only"); }
        const Q5_KBlock *blocks() const { return typed_data(); }

        Q5_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        /// Zero-copy constructor for mmap-backed data (no memcpy)
        Q5_KTensor(const std::vector<size_t> &shape, const uint8_t *mmap_data,
                   size_t byte_size, std::shared_ptr<void> mmap_lifetime_owner);
        ~Q5_KTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q5_K; }

        // IINT8Unpackable implementation
        void unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const override;
        float get_block_scale(size_t row_idx, size_t k_block_offset) const override;
        float get_block_min(size_t row_idx, size_t k_block_offset) const override;
        size_t superblock_size() const override { return 256; }
        const NativeVnniFormatInfo *vnniFormatInfo() const override
        {
            static constexpr NativeVnniFormatInfo info{7, 20, true, true, false, 31.0f};
            return &info;
        }
        void packVnniBlock(const VnniPackContext &ctx, int n, int b) const override;
        void unpack_superblock_to_int8(size_t row_idx, size_t superblock_idx, int8_t *output, float *scales = nullptr, float *mins = nullptr) const override;

        DeviceId home_device() const override { return device_; }

        const float *data() const override;
        float *mutable_data() override;

        // Diamond inheritance resolution (ITensor implemented by both TypedTensorBase and TensorBase)
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Memory management - release raw data after GEMM packing
        void release_raw_data() override
        {
            if (!is_view_ && !raw_data_released_)
            {
                raw_data_.clear();
                raw_data_.shrink_to_fit();
                raw_data_released_ = true;
            }
        }
        bool is_raw_data_released() const override { return raw_data_released_; }

        void release_host_weight_data() override
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
            {
                decltype(dequant_cache_) tmp;
                std::swap(dequant_cache_, tmp);
            }
            mmap_owner_.reset();
        }

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

    protected:
        // ===== Lazy Transfer Accessors (Phase 3) =====
        void *raw_host_data_ptr() override
        {
            return is_view_ ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        const void *raw_host_data_ptr() const override
        {
            return is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        size_t byte_size() const override { return data_byte_size_ > 0 ? data_byte_size_ : raw_data_.size(); }

    private:
        // View constructor (borrows parent's data)
        Q5_KTensor(
            const std::vector<size_t> &shape,
            const uint8_t *parent_raw_data,
            size_t byte_offset,
            std::shared_ptr<TensorBase> parent);

        std::vector<size_t> shape_;
        std::vector<uint8_t> raw_data_; // Owned only by parent tensor
        DeviceId device_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        bool raw_data_released_ = false; // Track if raw data was released after GEMM pack

        // View support fields
        bool is_view_;
        const uint8_t *raw_data_ptr_;        // Points to parent's raw_data_.data()
        size_t view_byte_offset_;            // Byte offset from raw_data_ptr_
        std::shared_ptr<TensorBase> parent_; // Keeps parent alive
        std::shared_ptr<void> mmap_owner_;   // Keeps mmap region alive for zero-copy tensors
        size_t data_byte_size_ = 0;          // Byte size for mmap-backed tensors (raw_data_ is empty)

        static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m);
    };

    // Implementation: Q3_KTensor.cpp
    /**
     * @brief Q3_K tensor (3-bit K-quant super-block)
     */
    class Q3_KTensor : public TypedTensorBase<Q3_KTensor, Q3_KBlock>, public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable, public IINT8Unpackable
    {
    public:
        /// Native storage type for CRTP-style type-safe access
        using value_type = Q3_KBlock;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::Q3_K; }

        // TypedTensorBase CRTP implementation
        const Q3_KBlock *data_impl() const { return reinterpret_cast<const Q3_KBlock *>(raw_host_data_ptr()); }
        Q3_KBlock *mutable_data_impl() { throw std::runtime_error("Q3_KTensor: weight tensors are read-only"); }
        const Q3_KBlock *blocks() const { return typed_data(); }

        Q3_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        /// Zero-copy constructor for mmap-backed data (no memcpy)
        Q3_KTensor(const std::vector<size_t> &shape, const uint8_t *mmap_data,
                   size_t byte_size, std::shared_ptr<void> mmap_lifetime_owner);
        ~Q3_KTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q3_K; }

        DeviceId home_device() const override { return device_; }

        const float *data() const override;
        float *mutable_data() override;

        // Diamond inheritance resolution (ITensor implemented by both TypedTensorBase and TensorBase)
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Memory management - release raw data after GEMM packing
        void release_raw_data() override
        {
            if (!is_view_ && !raw_data_released_)
            {
                raw_data_.clear();
                raw_data_.shrink_to_fit();
                raw_data_released_ = true;
            }
        }
        bool is_raw_data_released() const override { return raw_data_released_; }

        void release_host_weight_data() override
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
            {
                decltype(dequant_cache_) tmp;
                std::swap(dequant_cache_, tmp);
            }
            mmap_owner_.reset();
        }

        // IINT8Unpackable interface
        void unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const override;
        float get_block_scale(size_t row_idx, size_t k_block_offset) const override;
        float get_block_min(size_t row_idx, size_t k_block_offset) const override;
        size_t superblock_size() const override { return 256; }
        const NativeVnniFormatInfo *vnniFormatInfo() const override
        {
            static constexpr NativeVnniFormatInfo info{9, 12, true, true, false, 4.0f};
            return &info;
        }
        void packVnniBlock(const VnniPackContext &ctx, int n, int b) const override;
        void unpack_superblock_to_int8(size_t row_idx, size_t superblock_idx, int8_t *output, float *scales = nullptr, float *mins = nullptr) const override;

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

    protected:
        // ===== Lazy Transfer Accessors (Phase 3) =====
        void *raw_host_data_ptr() override
        {
            return is_view_ ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        const void *raw_host_data_ptr() const override
        {
            return is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        size_t byte_size() const override { return data_byte_size_ > 0 ? data_byte_size_ : raw_data_.size(); }

    private:
        std::vector<size_t> shape_;

        // Data ownership
        bool is_view_;
        std::vector<uint8_t> raw_data_;      // Owned data (if !is_view_)
        const uint8_t *raw_data_ptr_;        // Borrowed data (if is_view_)
        size_t view_byte_offset_;            // Byte offset in parent's raw_data_
        std::shared_ptr<TensorBase> parent_; // Keep parent alive (if is_view_)
        std::shared_ptr<void> mmap_owner_;   // Keeps mmap region alive for zero-copy tensors
        size_t data_byte_size_ = 0;          // Byte size for mmap-backed tensors (raw_data_ is empty)

        DeviceId device_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        bool raw_data_released_ = false; // Track if raw data was released after GEMM pack

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
    class Q4_KTensor : public TypedTensorBase<Q4_KTensor, Q4_KBlock>, public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable, public IINT8Unpackable
    {
    public:
        /// Native storage type for CRTP-style type-safe access
        using value_type = Q4_KBlock;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::Q4_K; }

        // TypedTensorBase CRTP implementation
        const Q4_KBlock *data_impl() const { return reinterpret_cast<const Q4_KBlock *>(raw_host_data_ptr()); }
        Q4_KBlock *mutable_data_impl() { throw std::runtime_error("Q4_KTensor: weight tensors are read-only"); }
        const Q4_KBlock *blocks() const { return typed_data(); }

        Q4_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        /// Zero-copy constructor for mmap-backed data (no memcpy)
        Q4_KTensor(const std::vector<size_t> &shape, const uint8_t *mmap_data,
                   size_t byte_size, std::shared_ptr<void> mmap_lifetime_owner);
        ~Q4_KTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q4_K; }

        DeviceId home_device() const override { return device_; }

        const float *data() const override;
        float *mutable_data() override;

        // Diamond inheritance resolution (ITensor implemented by both TypedTensorBase and TensorBase)
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Memory management - release raw data after GEMM packing
        void release_raw_data() override
        {
            if (!is_view_ && !raw_data_released_)
            {
                raw_data_.clear();
                raw_data_.shrink_to_fit();
                raw_data_released_ = true;
            }
        }
        bool is_raw_data_released() const override { return raw_data_released_; }

        void release_host_weight_data() override
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
            {
                decltype(dequant_cache_) tmp;
                std::swap(dequant_cache_, tmp);
            }
            mmap_owner_.reset();
        }

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

        // IINT8Unpackable interface
        void unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const override;
        float get_block_scale(size_t row_idx, size_t k_block_offset) const override;
        float get_block_min(size_t row_idx, size_t k_block_offset) const override;
        size_t superblock_size() const override { return 256; }
        const NativeVnniFormatInfo *vnniFormatInfo() const override
        {
            static constexpr NativeVnniFormatInfo info{5, 16, true, true, false, 15.0f};
            return &info;
        }
        void packVnniBlock(const VnniPackContext &ctx, int n, int b) const override;
        void unpack_superblock_to_int8(size_t row_idx, size_t superblock_idx, int8_t *output, float *scales = nullptr, float *mins = nullptr) const override;

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

    protected:
        // ===== Lazy Transfer Accessors (Phase 3) =====
        void *raw_host_data_ptr() override
        {
            return is_view_ ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        const void *raw_host_data_ptr() const override
        {
            return is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        size_t byte_size() const override { return data_byte_size_ > 0 ? data_byte_size_ : raw_data_.size(); }

    private:
        // View constructor (borrows parent's data)
        Q4_KTensor(
            const std::vector<size_t> &shape,
            const uint8_t *parent_raw_data,
            size_t byte_offset,
            std::shared_ptr<TensorBase> parent);

        std::vector<size_t> shape_;
        std::vector<uint8_t> raw_data_; // Owned only by parent tensor
        DeviceId device_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        bool raw_data_released_ = false; // Track if raw data was released after GEMM pack

        // View support fields
        bool is_view_;
        const uint8_t *raw_data_ptr_;        // Points to parent's raw_data_.data()
        size_t view_byte_offset_;            // Byte offset from raw_data_ptr_
        std::shared_ptr<TensorBase> parent_; // Keeps parent alive
        std::shared_ptr<void> mmap_owner_;   // Keeps mmap region alive for zero-copy tensors
        size_t data_byte_size_ = 0;          // Byte size for mmap-backed tensors (raw_data_ is empty)

        static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m);
    };

    // Implementation: Q8_KTensor.cpp
    /**
     * @brief Q8_K tensor (8-bit K-quant super-block)
     */
    class Q8_KTensor : public TypedTensorBase<Q8_KTensor, Q8_KBlock>, public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable, public IINT8Unpackable
    {
    public:
        /// Native storage type for CRTP-style type-safe access
        using value_type = Q8_KBlock;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::Q8_K; }

        // TypedTensorBase CRTP implementation
        const Q8_KBlock *data_impl() const { return reinterpret_cast<const Q8_KBlock *>(raw_host_data_ptr()); }
        Q8_KBlock *mutable_data_impl() { throw std::runtime_error("Q8_KTensor: weight tensors are read-only"); }
        const Q8_KBlock *blocks() const { return typed_data(); }

        Q8_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        /// Zero-copy constructor for mmap-backed data (no memcpy)
        Q8_KTensor(const std::vector<size_t> &shape, const uint8_t *mmap_data,
                   size_t byte_size, std::shared_ptr<void> mmap_lifetime_owner);
        ~Q8_KTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q8_K; }

        DeviceId home_device() const override { return device_; }

        const float *data() const override;
        float *mutable_data() override;

        // Diamond inheritance resolution (ITensor implemented by both TypedTensorBase and TensorBase)
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Memory management - release raw data after GEMM packing
        void release_raw_data() override
        {
            if (!is_view_ && !raw_data_released_)
            {
                raw_data_.clear();
                raw_data_.shrink_to_fit();
                raw_data_released_ = true;
            }
        }
        bool is_raw_data_released() const override { return raw_data_released_; }

        void release_host_weight_data() override
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
            {
                decltype(dequant_cache_) tmp;
                std::swap(dequant_cache_, tmp);
            }
            mmap_owner_.reset();
        }

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

        // IINT8Unpackable interface — Q8_K has no per-block scale (scale = 1.0f)
        void unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const override
        {
            // k_block_offset is in 32-element units; Q8_K uses 256-element super-blocks
            const size_t superblock_idx = k_block_offset / 8;
            const size_t sub_block = k_block_offset % 8;
            const Q8_KBlock *block = static_cast<const Q8_KBlock *>(get_raw_block_at(row_idx, superblock_idx));
            std::memcpy(output, block->qs + sub_block * 32, 32);
        }

        float get_block_scale(size_t row_idx, size_t k_block_offset) const override
        {
            (void)row_idx;
            (void)k_block_offset;
            return 1.0f; // Q8_K has no per-block scale
        }

        size_t superblock_size() const override { return 256; }

        /// Efficient override: reads Q8_K superblocks directly via typed_data().
        /// Q8_K has no per-block scale so values are already int8; only needs
        /// rescaling to maximize [-127,127] dynamic range.
        float requantizeRowToInt8(size_t row_idx, size_t K, int8_t *output) const override;

        void unpack_superblock_to_int8(
            size_t row_idx, size_t superblock_idx,
            int8_t *output, float *scales = nullptr, float *mins = nullptr) const override
        {
            const Q8_KBlock *block = static_cast<const Q8_KBlock *>(get_raw_block_at(row_idx, superblock_idx));
            std::memcpy(output, block->qs, 256);
            if (scales)
            {
                for (int i = 0; i < 8; ++i)
                    scales[i] = 1.0f;
            }
            if (mins)
            {
                for (int i = 0; i < 8; ++i)
                    mins[i] = 0.0f;
            }
        }

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

    protected:
        // ===== Lazy Transfer Accessors (Phase 3) =====
        void *raw_host_data_ptr() override
        {
            return is_view_ ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        const void *raw_host_data_ptr() const override
        {
            return is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        size_t byte_size() const override { return data_byte_size_ > 0 ? data_byte_size_ : raw_data_.size(); }

    private:
        // View constructor (borrows parent's data)
        Q8_KTensor(
            const std::vector<size_t> &shape,
            const uint8_t *parent_raw_data,
            size_t byte_offset,
            std::shared_ptr<TensorBase> parent);

        std::vector<size_t> shape_;
        std::vector<uint8_t> raw_data_; // Owned only by parent tensor
        DeviceId device_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        bool raw_data_released_ = false; // Track if raw data was released after GEMM pack

        // View support fields
        bool is_view_;
        const uint8_t *raw_data_ptr_;        // Points to parent's raw_data_.data()
        size_t view_byte_offset_;            // Byte offset from raw_data_ptr_
        std::shared_ptr<TensorBase> parent_; // Keeps parent alive
        std::shared_ptr<void> mmap_owner_;   // Keeps mmap region alive for zero-copy tensors
        size_t data_byte_size_ = 0;          // Byte size for mmap-backed tensors (raw_data_ is empty)
    };

    // ===== IQ Tensors =====

    // Implementation: IQ4_XSTensor.cpp
    /**
     * @brief IQ4_XS tensor (4-bit extra-small IQ)
     */
    class IQ4_XSTensor : public TypedTensorBase<IQ4_XSTensor, IQ4_XSBlock>, public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable, public IINT8Unpackable
    {
    public:
        /// Native storage type for CRTP-style type-safe access
        using value_type = IQ4_XSBlock;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::IQ4_XS; }

        // TypedTensorBase CRTP implementation
        const IQ4_XSBlock *data_impl() const { return reinterpret_cast<const IQ4_XSBlock *>(raw_host_data_ptr()); }
        IQ4_XSBlock *mutable_data_impl() { throw std::runtime_error("IQ4_XSTensor: weight tensors are read-only"); }
        const IQ4_XSBlock *blocks() const { return typed_data(); }

        IQ4_XSTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        /// Zero-copy constructor for mmap-backed data (no memcpy)
        IQ4_XSTensor(const std::vector<size_t> &shape, const uint8_t *mmap_data,
                     size_t byte_size, std::shared_ptr<void> mmap_lifetime_owner);
        ~IQ4_XSTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ4_XS; }

        DeviceId home_device() const override { return device_; }

        const float *data() const override;
        float *mutable_data() override;

        // Diamond inheritance resolution (ITensor implemented by both TypedTensorBase and TensorBase)
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Memory management - release raw data after GEMM packing
        void release_raw_data() override
        {
            if (!is_view_ && !raw_data_released_)
            {
                raw_data_.clear();
                raw_data_.shrink_to_fit();
                raw_data_released_ = true;
            }
        }
        bool is_raw_data_released() const override { return raw_data_released_; }

        void release_host_weight_data() override
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
            {
                decltype(dequant_cache_) tmp;
                std::swap(dequant_cache_, tmp);
            }
            mmap_owner_.reset();
        }

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
            const size_t super_blocks_per_row = (shape_[1] + IQ4_XSBlock::BLOCK_SIZE - 1) / IQ4_XSBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ4_XSBlock *blocks = reinterpret_cast<const IQ4_XSBlock *>(data_ptr);

            const size_t super_block_idx = (row_idx * super_blocks_per_row) + (k_block_offset / 8);
            const size_t sub_block_idx = k_block_offset % 8;

            int8_t temp_int8[32];
            simd::unpack_iq4_xs_to_int8(blocks[super_block_idx], sub_block_idx, temp_int8);
            float scale = simd::get_iq4_xs_scale(blocks[super_block_idx], sub_block_idx);

            for (int i = 0; i < 32; ++i)
            {
                output[i] = temp_int8[i] * scale;
            }
        }

        __attribute__((always_inline))
        const void *
        get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t super_blocks_per_row = (shape_[1] + IQ4_XSBlock::BLOCK_SIZE - 1) / IQ4_XSBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ4_XSBlock *blocks = reinterpret_cast<const IQ4_XSBlock *>(data_ptr);

            const size_t super_block_idx = (row_idx * super_blocks_per_row) + (k_block_offset / 8);

            return &blocks[super_block_idx];
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return 32; } // Logical block size (sub-block)

        // IINT8Unpackable interface
        __attribute__((always_inline)) void unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const override
        {
            const size_t super_blocks_per_row = (shape_[1] + IQ4_XSBlock::BLOCK_SIZE - 1) / IQ4_XSBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ4_XSBlock *blocks = reinterpret_cast<const IQ4_XSBlock *>(data_ptr);

            const size_t super_block_idx = (row_idx * super_blocks_per_row) + (k_block_offset / 8);
            const size_t sub_block_idx = k_block_offset % 8;

            simd::unpack_iq4_xs_to_int8(blocks[super_block_idx], sub_block_idx, output);
        }

        __attribute__((always_inline)) float get_block_scale(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t super_blocks_per_row = (shape_[1] + IQ4_XSBlock::BLOCK_SIZE - 1) / IQ4_XSBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ4_XSBlock *blocks = reinterpret_cast<const IQ4_XSBlock *>(data_ptr);

            const size_t super_block_idx = (row_idx * super_blocks_per_row) + (k_block_offset / 8);
            const size_t sub_block_idx = k_block_offset % 8;

            return simd::get_iq4_xs_scale(blocks[super_block_idx], sub_block_idx);
        }

        size_t superblock_size() const override { return 256; }
        const NativeVnniFormatInfo *vnniFormatInfo() const override
        {
            static constexpr NativeVnniFormatInfo info{4, 16, false, true, false, 127.0f};
            return &info;
        }
        void packVnniBlock(const VnniPackContext &ctx, int n, int b) const override;
        void unpack_superblock_to_int8(size_t row_idx, size_t superblock_idx, int8_t *output, float *scales = nullptr, float *mins = nullptr) const override;

        // SIMD decode methods (public for testing)
        static void decodeBlock(const IQ4_XSBlock &block, float *output);
        static void decodeBlockScalar(const IQ4_XSBlock &block, float *output);
#ifdef __AVX512F__
        static void decodeBlockAVX512(const IQ4_XSBlock &block, float *output);
#endif
#ifdef __AVX2__
        static void decodeBlockAVX2(const IQ4_XSBlock &block, float *output);
#endif

    protected:
        // ===== Lazy Transfer Accessors (Phase 3) =====
        void *raw_host_data_ptr() override
        {
            return is_view_ ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        const void *raw_host_data_ptr() const override
        {
            return is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        size_t byte_size() const override { return data_byte_size_ > 0 ? data_byte_size_ : raw_data_.size(); }

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
        std::shared_ptr<void> mmap_owner_;   // Keeps mmap region alive for zero-copy tensors
        size_t data_byte_size_ = 0;          // Byte size for mmap-backed tensors (raw_data_ is empty)

        DeviceId device_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        bool raw_data_released_ = false; // Track if raw data was released after GEMM pack
    };

    // Implementation: IQ2_XXSTensor.cpp
    /**
     * @brief IQ2_XXS tensor (2-bit extra-extra-small IQ)
     */
    class IQ2_XXSTensor : public TypedTensorBase<IQ2_XXSTensor, IQ2_XXSBlock>, public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable, public IINT8Unpackable
    {
    public:
        /// Native storage type for CRTP-style type-safe access
        using value_type = IQ2_XXSBlock;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::IQ2_XXS; }

        // TypedTensorBase CRTP implementation
        const IQ2_XXSBlock *data_impl() const { return reinterpret_cast<const IQ2_XXSBlock *>(raw_host_data_ptr()); }
        IQ2_XXSBlock *mutable_data_impl() { throw std::runtime_error("IQ2_XXSTensor: weight tensors are read-only"); }
        const IQ2_XXSBlock *blocks() const { return typed_data(); }

        IQ2_XXSTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        /// Zero-copy constructor for mmap-backed data (no memcpy)
        IQ2_XXSTensor(const std::vector<size_t> &shape, const uint8_t *mmap_data,
                      size_t byte_size, std::shared_ptr<void> mmap_lifetime_owner);
        ~IQ2_XXSTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ2_XXS; }

        DeviceId home_device() const override { return device_; }

        const float *data() const override;
        float *mutable_data() override;

        // Diamond inheritance resolution (ITensor implemented by both TypedTensorBase and TensorBase)
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Memory management - release raw data after GEMM packing
        void release_raw_data() override
        {
            if (!is_view_ && !raw_data_released_)
            {
                raw_data_.clear();
                raw_data_.shrink_to_fit();
                raw_data_released_ = true;
            }
        }
        bool is_raw_data_released() const override { return raw_data_released_; }

        void release_host_weight_data() override
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
            {
                decltype(dequant_cache_) tmp;
                std::swap(dequant_cache_, tmp);
            }
            mmap_owner_.reset();
        }

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

        // IINT8Unpackable interface
        __attribute__((always_inline)) void unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const override
        {
            const size_t super_blocks_per_row = (shape_[1] + IQ2_XXSBlock::BLOCK_SIZE - 1) / IQ2_XXSBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ2_XXSBlock *blocks = reinterpret_cast<const IQ2_XXSBlock *>(data_ptr);

            const size_t super_block_idx = (row_idx * super_blocks_per_row) + (k_block_offset / 8);
            const size_t sub_block_idx = k_block_offset % 8;

            simd::unpack_iq2_xxs_to_int8(blocks[super_block_idx], sub_block_idx, output);
        }

        __attribute__((always_inline)) float get_block_scale(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t super_blocks_per_row = (shape_[1] + IQ2_XXSBlock::BLOCK_SIZE - 1) / IQ2_XXSBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ2_XXSBlock *blocks = reinterpret_cast<const IQ2_XXSBlock *>(data_ptr);

            const size_t super_block_idx = (row_idx * super_blocks_per_row) + (k_block_offset / 8);
            const size_t sub_block_idx = k_block_offset % 8;

            return simd::get_iq2_xxs_scale(blocks[super_block_idx], sub_block_idx);
        }

        size_t superblock_size() const override { return 256; }
        const NativeVnniFormatInfo *vnniFormatInfo() const override
        {
            static constexpr NativeVnniFormatInfo info{15, 8, false, true, false, 43.0f};
            return &info;
        }
        void packVnniBlock(const VnniPackContext &ctx, int n, int b) const override;
        void unpack_superblock_to_int8(size_t row_idx, size_t superblock_idx, int8_t *output, float *scales = nullptr, float *mins = nullptr) const override;

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

    protected:
        // ===== Lazy Transfer Accessors (Phase 3) =====
        void *raw_host_data_ptr() override
        {
            return is_view_ ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        const void *raw_host_data_ptr() const override
        {
            return is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        size_t byte_size() const override { return data_byte_size_ > 0 ? data_byte_size_ : raw_data_.size(); }

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
        std::shared_ptr<void> mmap_owner_;   // Keeps mmap region alive for zero-copy tensors
        size_t data_byte_size_ = 0;          // Byte size for mmap-backed tensors (raw_data_ is empty)

        DeviceId device_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        bool raw_data_released_ = false; // Track if raw data was released after GEMM pack
    };

    // Implementation: IQ2_XSTensor.cpp
    /**
     * @brief IQ2_XS tensor (2-bit extra-small IQ)
     */
    class IQ2_XSTensor : public TypedTensorBase<IQ2_XSTensor, IQ2_XSBlock>, public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable, public IINT8Unpackable
    {
    public:
        /// Native storage type for CRTP-style type-safe access
        using value_type = IQ2_XSBlock;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::IQ2_XS; }

        // TypedTensorBase CRTP implementation
        const IQ2_XSBlock *data_impl() const { return reinterpret_cast<const IQ2_XSBlock *>(raw_host_data_ptr()); }
        IQ2_XSBlock *mutable_data_impl() { throw std::runtime_error("IQ2_XSTensor: weight tensors are read-only"); }
        const IQ2_XSBlock *blocks() const { return typed_data(); }

        IQ2_XSTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        /// Zero-copy constructor for mmap-backed data (no memcpy)
        IQ2_XSTensor(const std::vector<size_t> &shape, const uint8_t *mmap_data,
                     size_t byte_size, std::shared_ptr<void> mmap_lifetime_owner);
        ~IQ2_XSTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ2_XS; }

        DeviceId home_device() const override { return device_; }

        const float *data() const override;
        float *mutable_data() override;

        // Diamond inheritance resolution (ITensor implemented by both TypedTensorBase and TensorBase)
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Memory management - release raw data after GEMM packing
        void release_raw_data() override
        {
            if (!is_view_ && !raw_data_released_)
            {
                raw_data_.clear();
                raw_data_.shrink_to_fit();
                raw_data_released_ = true;
            }
        }
        bool is_raw_data_released() const override { return raw_data_released_; }

        void release_host_weight_data() override
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
            {
                decltype(dequant_cache_) tmp;
                std::swap(dequant_cache_, tmp);
            }
            mmap_owner_.reset();
        }

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

        // IINT8Unpackable interface
        __attribute__((always_inline)) void unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const override
        {
            const size_t super_blocks_per_row = (shape_[1] + IQ2_XSBlock::BLOCK_SIZE - 1) / IQ2_XSBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ2_XSBlock *blocks = reinterpret_cast<const IQ2_XSBlock *>(data_ptr);

            const size_t super_block_idx = (row_idx * super_blocks_per_row) + (k_block_offset / 8);
            const size_t sub_block_idx = k_block_offset % 8;

            simd::unpack_iq2_xs_to_int8(blocks[super_block_idx], sub_block_idx, output);
        }

        __attribute__((always_inline)) float get_block_scale(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t super_blocks_per_row = (shape_[1] + IQ2_XSBlock::BLOCK_SIZE - 1) / IQ2_XSBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ2_XSBlock *blocks = reinterpret_cast<const IQ2_XSBlock *>(data_ptr);

            const size_t super_block_idx = (row_idx * super_blocks_per_row) + (k_block_offset / 8);
            const size_t sub_block_idx = k_block_offset % 8;

            return simd::get_iq2_xs_scale(blocks[super_block_idx], sub_block_idx);
        }

        size_t superblock_size() const override { return 256; }
        const NativeVnniFormatInfo *vnniFormatInfo() const override
        {
            static constexpr NativeVnniFormatInfo info{14, 9, true, true, false, 43.0f};
            return &info;
        }
        void packVnniBlock(const VnniPackContext &ctx, int n, int b) const override;
        void unpack_superblock_to_int8(size_t row_idx, size_t superblock_idx, int8_t *output, float *scales = nullptr, float *mins = nullptr) const override;

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

    protected:
        // ===== Lazy Transfer Accessors (Phase 3) =====
        void *raw_host_data_ptr() override
        {
            return is_view_ ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        const void *raw_host_data_ptr() const override
        {
            return is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        size_t byte_size() const override { return data_byte_size_ > 0 ? data_byte_size_ : raw_data_.size(); }

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
        std::shared_ptr<void> mmap_owner_;   // Keeps mmap region alive for zero-copy tensors
        size_t data_byte_size_ = 0;          // Byte size for mmap-backed tensors (raw_data_ is empty)

        DeviceId device_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        bool raw_data_released_ = false; // Track if raw data was released after GEMM pack
    };

    // Implementation: IQ3_XXSTensor.cpp
    /**
     * @brief IQ3_XXS tensor (3-bit extra-extra-small IQ)
     */
    class IQ3_XXSTensor : public TypedTensorBase<IQ3_XXSTensor, IQ3_XXSBlock>, public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable, public IINT8Unpackable
    {
    public:
        /// Native storage type for CRTP-style type-safe access
        using value_type = IQ3_XXSBlock;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::IQ3_XXS; }

        // TypedTensorBase CRTP implementation
        const IQ3_XXSBlock *data_impl() const { return reinterpret_cast<const IQ3_XXSBlock *>(raw_host_data_ptr()); }
        IQ3_XXSBlock *mutable_data_impl() { throw std::runtime_error("IQ3_XXSTensor: weight tensors are read-only"); }
        const IQ3_XXSBlock *blocks() const { return typed_data(); }

        IQ3_XXSTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        /// Zero-copy constructor for mmap-backed data (no memcpy)
        IQ3_XXSTensor(const std::vector<size_t> &shape, const uint8_t *mmap_data,
                      size_t byte_size, std::shared_ptr<void> mmap_lifetime_owner);
        ~IQ3_XXSTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ3_XXS; }

        DeviceId home_device() const override { return device_; }

        const float *data() const override;
        float *mutable_data() override;

        // Diamond inheritance resolution (ITensor implemented by both TypedTensorBase and TensorBase)
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Memory management - release raw data after GEMM packing
        void release_raw_data() override
        {
            if (!is_view_ && !raw_data_released_)
            {
                raw_data_.clear();
                raw_data_.shrink_to_fit();
                raw_data_released_ = true;
            }
        }
        bool is_raw_data_released() const override { return raw_data_released_; }

        void release_host_weight_data() override
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
            {
                decltype(dequant_cache_) tmp;
                std::swap(dequant_cache_, tmp);
            }
            mmap_owner_.reset();
        }

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
            const size_t super_blocks_per_row = (shape_[1] + IQ3_XXSBlock::BLOCK_SIZE - 1) / IQ3_XXSBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ3_XXSBlock *blocks = reinterpret_cast<const IQ3_XXSBlock *>(data_ptr);

            const size_t super_block_idx = (row_idx * super_blocks_per_row) + (k_block_offset / 8);
            const size_t sub_block_idx = k_block_offset % 8;

            simd::decode_iq3xxs_subblock_to_fp32(blocks[super_block_idx], sub_block_idx, output);
        }

        __attribute__((always_inline))
        const void *
        get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t super_blocks_per_row = (shape_[1] + IQ3_XXSBlock::BLOCK_SIZE - 1) / IQ3_XXSBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ3_XXSBlock *blocks = reinterpret_cast<const IQ3_XXSBlock *>(data_ptr);

            const size_t super_block_idx = (row_idx * super_blocks_per_row) + (k_block_offset / 8);
            return &blocks[super_block_idx];
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return 32; } // Logical block size (sub-block)

        // IINT8Unpackable interface
        __attribute__((always_inline)) void unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const override
        {
            const size_t super_blocks_per_row = (shape_[1] + IQ3_XXSBlock::BLOCK_SIZE - 1) / IQ3_XXSBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ3_XXSBlock *blocks = reinterpret_cast<const IQ3_XXSBlock *>(data_ptr);

            const size_t super_block_idx = (row_idx * super_blocks_per_row) + (k_block_offset / 8);
            const size_t sub_block_idx = k_block_offset % 8;

            simd::unpack_iq3_xxs_to_int8(blocks[super_block_idx], sub_block_idx, output);
        }

        __attribute__((always_inline)) float get_block_scale(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t super_blocks_per_row = (shape_[1] + IQ3_XXSBlock::BLOCK_SIZE - 1) / IQ3_XXSBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ3_XXSBlock *blocks = reinterpret_cast<const IQ3_XXSBlock *>(data_ptr);

            const size_t super_block_idx = (row_idx * super_blocks_per_row) + (k_block_offset / 8);
            const size_t sub_block_idx = k_block_offset % 8;

            return simd::get_iq3_xxs_scale(blocks[super_block_idx], sub_block_idx);
        }

        size_t superblock_size() const override { return 256; }
        const NativeVnniFormatInfo *vnniFormatInfo() const override
        {
            static constexpr NativeVnniFormatInfo info{12, 12, false, true, false, 62.0f};
            return &info;
        }
        void packVnniBlock(const VnniPackContext &ctx, int n, int b) const override;
        void unpack_superblock_to_int8(size_t row_idx, size_t superblock_idx, int8_t *output, float *scales = nullptr, float *mins = nullptr) const override;

        // SIMD decode methods (public for testing)
        static void decodeBlock(const IQ3_XXSBlock &block, float *output);
        static void decodeBlockScalar(const IQ3_XXSBlock &block, float *output);
#ifdef __AVX512F__
        static void decodeBlockAVX512(const IQ3_XXSBlock &block, float *output);
#endif
#ifdef __AVX2__
        static void decodeBlockAVX2(const IQ3_XXSBlock &block, float *output);
#endif

    protected:
        // ===== Lazy Transfer Accessors (Phase 3) =====
        void *raw_host_data_ptr() override
        {
            return is_view_ ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        const void *raw_host_data_ptr() const override
        {
            return is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        size_t byte_size() const override { return data_byte_size_ > 0 ? data_byte_size_ : raw_data_.size(); }

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
        std::shared_ptr<void> mmap_owner_;   // Keeps mmap region alive for zero-copy tensors
        size_t data_byte_size_ = 0;          // Byte size for mmap-backed tensors (raw_data_ is empty)

        DeviceId device_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        bool raw_data_released_ = false; // Track if raw data was released after GEMM pack
    };

    // Implementation: IQ2_STensor.cpp
    /**
     * @brief IQ2_S tensor (2-bit small IQ)
     */
    class IQ2_STensor : public TypedTensorBase<IQ2_STensor, IQ2_SBlock>, public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable, public IINT8Unpackable
    {
    public:
        /// Native storage type for CRTP-style type-safe access
        using value_type = IQ2_SBlock;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::IQ2_S; }

        // TypedTensorBase CRTP implementation
        const IQ2_SBlock *data_impl() const { return reinterpret_cast<const IQ2_SBlock *>(raw_host_data_ptr()); }
        IQ2_SBlock *mutable_data_impl() { throw std::runtime_error("IQ2_STensor: weight tensors are read-only"); }
        const IQ2_SBlock *blocks() const { return typed_data(); }

        IQ2_STensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        /// Zero-copy constructor for mmap-backed data (no memcpy)
        IQ2_STensor(const std::vector<size_t> &shape, const uint8_t *mmap_data,
                    size_t byte_size, std::shared_ptr<void> mmap_lifetime_owner);
        ~IQ2_STensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ2_S; }

        DeviceId home_device() const override { return device_; }

        const float *data() const override;
        float *mutable_data() override;

        // Diamond inheritance resolution (ITensor implemented by both TypedTensorBase and TensorBase)
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Memory management - release raw data after GEMM packing
        void release_raw_data() override
        {
            if (!is_view_ && !raw_data_released_)
            {
                raw_data_.clear();
                raw_data_.shrink_to_fit();
                raw_data_released_ = true;
            }
        }
        bool is_raw_data_released() const override { return raw_data_released_; }

        void release_host_weight_data() override
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
            {
                decltype(dequant_cache_) tmp;
                std::swap(dequant_cache_, tmp);
            }
            mmap_owner_.reset();
        }

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

        // IINT8Unpackable interface
        __attribute__((always_inline)) void unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const override
        {
            const size_t super_blocks_per_row = (shape_[1] + IQ2_SBlock::BLOCK_SIZE - 1) / IQ2_SBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ2_SBlock *blocks = reinterpret_cast<const IQ2_SBlock *>(data_ptr);

            const size_t super_block_idx = (row_idx * super_blocks_per_row) + (k_block_offset / 8);
            const size_t sub_block_idx = k_block_offset % 8;

            simd::unpack_iq2_s_to_int8(blocks[super_block_idx], sub_block_idx, output);
        }

        __attribute__((always_inline)) float get_block_scale(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t super_blocks_per_row = (shape_[1] + IQ2_SBlock::BLOCK_SIZE - 1) / IQ2_SBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ2_SBlock *blocks = reinterpret_cast<const IQ2_SBlock *>(data_ptr);

            const size_t super_block_idx = (row_idx * super_blocks_per_row) + (k_block_offset / 8);
            const size_t sub_block_idx = k_block_offset % 8;

            return simd::get_iq2_s_scale(blocks[super_block_idx], sub_block_idx);
        }

        size_t superblock_size() const override { return 256; }
        const NativeVnniFormatInfo *vnniFormatInfo() const override
        {
            static constexpr NativeVnniFormatInfo info{13, 9, true, true, false, 43.0f};
            return &info;
        }
        void packVnniBlock(const VnniPackContext &ctx, int n, int b) const override;
        void unpack_superblock_to_int8(size_t row_idx, size_t superblock_idx, int8_t *output, float *scales = nullptr, float *mins = nullptr) const override;

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

    protected:
        // ===== Lazy Transfer Accessors (Phase 3) =====
        void *raw_host_data_ptr() override
        {
            return is_view_ ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        const void *raw_host_data_ptr() const override
        {
            return is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        size_t byte_size() const override { return data_byte_size_ > 0 ? data_byte_size_ : raw_data_.size(); }

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
        std::shared_ptr<void> mmap_owner_;   // Keeps mmap region alive for zero-copy tensors
        size_t data_byte_size_ = 0;          // Byte size for mmap-backed tensors (raw_data_ is empty)

        DeviceId device_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        bool raw_data_released_ = false; // Track if raw data was released after GEMM pack
    };

    // Implementation: IQ3_STensor.cpp
    /**
     * @brief IQ3_S tensor (3-bit small IQ)
     */
    class IQ3_STensor : public TypedTensorBase<IQ3_STensor, IQ3_SBlock>, public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable, public IINT8Unpackable
    {
    public:
        /// Native storage type for CRTP-style type-safe access
        using value_type = IQ3_SBlock;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::IQ3_S; }

        // TypedTensorBase CRTP implementation
        const IQ3_SBlock *data_impl() const { return reinterpret_cast<const IQ3_SBlock *>(raw_host_data_ptr()); }
        IQ3_SBlock *mutable_data_impl() { throw std::runtime_error("IQ3_STensor: weight tensors are read-only"); }
        const IQ3_SBlock *blocks() const { return typed_data(); }

        IQ3_STensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        /// Zero-copy constructor for mmap-backed data (no memcpy)
        IQ3_STensor(const std::vector<size_t> &shape, const uint8_t *mmap_data,
                    size_t byte_size, std::shared_ptr<void> mmap_lifetime_owner);
        ~IQ3_STensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ3_S; }

        DeviceId home_device() const override { return device_; }

        const float *data() const override;
        float *mutable_data() override;

        // Diamond inheritance resolution (ITensor implemented by both TypedTensorBase and TensorBase)
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Memory management - release raw data after GEMM packing
        void release_raw_data() override
        {
            if (!is_view_ && !raw_data_released_)
            {
                raw_data_.clear();
                raw_data_.shrink_to_fit();
                raw_data_released_ = true;
            }
        }
        bool is_raw_data_released() const override { return raw_data_released_; }

        void release_host_weight_data() override
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
            {
                decltype(dequant_cache_) tmp;
                std::swap(dequant_cache_, tmp);
            }
            mmap_owner_.reset();
        }

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
            const size_t super_blocks_per_row = (shape_[1] + IQ3_SBlock::BLOCK_SIZE - 1) / IQ3_SBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ3_SBlock *blocks = reinterpret_cast<const IQ3_SBlock *>(data_ptr);

            const size_t super_block_idx = (row_idx * super_blocks_per_row) + (k_block_offset / 8);
            const size_t sub_block_idx = k_block_offset % 8;

            simd::decode_iq3s_subblock_to_fp32(blocks[super_block_idx], sub_block_idx, output);
        }

        __attribute__((always_inline))
        const void *
        get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t super_blocks_per_row = (shape_[1] + IQ3_SBlock::BLOCK_SIZE - 1) / IQ3_SBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ3_SBlock *blocks = reinterpret_cast<const IQ3_SBlock *>(data_ptr);

            const size_t super_block_idx = (row_idx * super_blocks_per_row) + (k_block_offset / 8);
            return &blocks[super_block_idx];
        }

        // IINT8Unpackable interface
        __attribute__((always_inline)) void unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const override
        {
            const size_t super_blocks_per_row = (shape_[1] + IQ3_SBlock::BLOCK_SIZE - 1) / IQ3_SBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ3_SBlock *blocks = reinterpret_cast<const IQ3_SBlock *>(data_ptr);

            const size_t super_block_idx = (row_idx * super_blocks_per_row) + (k_block_offset / 8);
            const size_t sub_block_idx = k_block_offset % 8;

            simd::unpack_iq3_s_to_int8(blocks[super_block_idx], sub_block_idx, output);
        }

        __attribute__((always_inline)) float get_block_scale(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t super_blocks_per_row = (shape_[1] + IQ3_SBlock::BLOCK_SIZE - 1) / IQ3_SBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ3_SBlock *blocks = reinterpret_cast<const IQ3_SBlock *>(data_ptr);

            const size_t super_block_idx = (row_idx * super_blocks_per_row) + (k_block_offset / 8);
            const size_t sub_block_idx = k_block_offset % 8;

            return simd::get_iq3_s_scale(blocks[super_block_idx], sub_block_idx);
        }

        size_t superblock_size() const override { return 256; }
        const NativeVnniFormatInfo *vnniFormatInfo() const override
        {
            static constexpr NativeVnniFormatInfo info{11, 13, false, true, false, 15.0f};
            return &info;
        }
        void packVnniBlock(const VnniPackContext &ctx, int n, int b) const override;
        void unpack_superblock_to_int8(size_t row_idx, size_t superblock_idx, int8_t *output, float *scales = nullptr, float *mins = nullptr) const override;

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return 32; }

        // SIMD decode methods (public for testing)
        static void decodeBlock(const IQ3_SBlock &block, float *output);
        static void decodeBlockScalar(const IQ3_SBlock &block, float *output);
#ifdef __AVX512F__
        static void decodeBlockAVX512(const IQ3_SBlock &block, float *output);
#endif
#ifdef __AVX2__
        static void decodeBlockAVX2(const IQ3_SBlock &block, float *output);
#endif

    protected:
        // ===== Lazy Transfer Accessors (Phase 3) =====
        void *raw_host_data_ptr() override
        {
            return is_view_ ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        const void *raw_host_data_ptr() const override
        {
            return is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        size_t byte_size() const override { return data_byte_size_ > 0 ? data_byte_size_ : raw_data_.size(); }

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
        std::shared_ptr<void> mmap_owner_;   // Keeps mmap region alive for zero-copy tensors
        size_t data_byte_size_ = 0;          // Byte size for mmap-backed tensors (raw_data_ is empty)

        DeviceId device_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        bool raw_data_released_ = false; // Track if raw data was released after GEMM pack
    };

    // Implementation: IQ1_STensor.cpp
    /**
     * @brief IQ1_S tensor (1-bit small IQ)
     */
    class IQ1_STensor : public TypedTensorBase<IQ1_STensor, IQ1_SBlock>, public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable, public IINT8Unpackable
    {
    public:
        /// Native storage type for CRTP-style type-safe access
        using value_type = IQ1_SBlock;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::IQ1_S; }

        // TypedTensorBase CRTP implementation
        const IQ1_SBlock *data_impl() const { return reinterpret_cast<const IQ1_SBlock *>(raw_host_data_ptr()); }
        IQ1_SBlock *mutable_data_impl() { throw std::runtime_error("IQ1_STensor: weight tensors are read-only"); }
        const IQ1_SBlock *blocks() const { return typed_data(); }

        IQ1_STensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        /// Zero-copy constructor for mmap-backed data (no memcpy)
        IQ1_STensor(const std::vector<size_t> &shape, const uint8_t *mmap_data,
                    size_t byte_size, std::shared_ptr<void> mmap_lifetime_owner);
        ~IQ1_STensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ1_S; }

        DeviceId home_device() const override { return device_; }

        const float *data() const override;
        float *mutable_data() override;

        // Diamond inheritance resolution (ITensor implemented by both TypedTensorBase and TensorBase)
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Memory management - release raw data after GEMM packing
        void release_raw_data() override
        {
            if (!is_view_ && !raw_data_released_)
            {
                raw_data_.clear();
                raw_data_.shrink_to_fit();
                raw_data_released_ = true;
            }
        }
        bool is_raw_data_released() const override { return raw_data_released_; }

        void release_host_weight_data() override
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
            {
                decltype(dequant_cache_) tmp;
                std::swap(dequant_cache_, tmp);
            }
            mmap_owner_.reset();
        }

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

        // IINT8Unpackable interface
        __attribute__((always_inline)) void unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const override
        {
            const size_t super_blocks_per_row = (shape_[1] + IQ1_SBlock::BLOCK_SIZE - 1) / IQ1_SBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ1_SBlock *blocks = reinterpret_cast<const IQ1_SBlock *>(data_ptr);

            const size_t super_block_idx = (row_idx * super_blocks_per_row) + (k_block_offset / 8);
            const size_t sub_block_idx = k_block_offset % 8;

            simd::unpack_iq1_s_to_int8(blocks[super_block_idx], sub_block_idx, output);
        }

        __attribute__((always_inline)) float get_block_scale(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t super_blocks_per_row = (shape_[1] + IQ1_SBlock::BLOCK_SIZE - 1) / IQ1_SBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ1_SBlock *blocks = reinterpret_cast<const IQ1_SBlock *>(data_ptr);

            const size_t super_block_idx = (row_idx * super_blocks_per_row) + (k_block_offset / 8);
            const size_t sub_block_idx = k_block_offset % 8;

            return simd::get_iq1_s_scale(blocks[super_block_idx], sub_block_idx);
        }

        size_t superblock_size() const override { return 256; }
        const NativeVnniFormatInfo *vnniFormatInfo() const override
        {
            static constexpr NativeVnniFormatInfo info{16, 6, true, true, false, 1.125f};
            return &info;
        }
        void packVnniBlock(const VnniPackContext &ctx, int n, int b) const override;
        void unpack_superblock_to_int8(size_t row_idx, size_t superblock_idx, int8_t *output, float *scales = nullptr, float *mins = nullptr) const override;

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

    protected:
        // ===== Lazy Transfer Accessors (Phase 3) =====
        void *raw_host_data_ptr() override
        {
            return is_view_ ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        const void *raw_host_data_ptr() const override
        {
            return is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        size_t byte_size() const override { return data_byte_size_ > 0 ? data_byte_size_ : raw_data_.size(); }

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
        std::shared_ptr<void> mmap_owner_;   // Keeps mmap region alive for zero-copy tensors
        size_t data_byte_size_ = 0;          // Byte size for mmap-backed tensors (raw_data_ is empty)

        DeviceId device_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        bool raw_data_released_ = false; // Track if raw data was released after GEMM pack
    };

    // Implementation: IQ1_MTensor.cpp
    /**
     * @brief IQ1_M tensor (1-bit medium IQ)
     */
    class IQ1_MTensor : public TypedTensorBase<IQ1_MTensor, IQ1_MBlock>, public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable, public IINT8Unpackable
    {
    public:
        /// Native storage type for CRTP-style type-safe access
        using value_type = IQ1_MBlock;

        /// Static type ID for ITensor::is<T>() and typed_as<T>()
        static constexpr int static_type_id() { return TensorTypeId::IQ1_M; }

        // TypedTensorBase CRTP implementation
        const IQ1_MBlock *data_impl() const { return reinterpret_cast<const IQ1_MBlock *>(raw_host_data_ptr()); }
        IQ1_MBlock *mutable_data_impl() { throw std::runtime_error("IQ1_MTensor: weight tensors are read-only"); }
        const IQ1_MBlock *blocks() const { return typed_data(); }

        IQ1_MTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        /// Zero-copy constructor for mmap-backed data (no memcpy)
        IQ1_MTensor(const std::vector<size_t> &shape, const uint8_t *mmap_data,
                    size_t byte_size, std::shared_ptr<void> mmap_lifetime_owner);
        ~IQ1_MTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ1_M; }

        DeviceId home_device() const override { return device_; }

        const float *data() const override;
        float *mutable_data() override;

        // Diamond inheritance resolution (ITensor implemented by both TypedTensorBase and TensorBase)
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return TensorBase::raw_data(); }
        void *raw_mutable_data() final { return TensorBase::raw_mutable_data(); }

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;

        // Memory management - release raw data after GEMM packing
        void release_raw_data() override
        {
            if (!is_view_ && !raw_data_released_)
            {
                raw_data_.clear();
                raw_data_.shrink_to_fit();
                raw_data_released_ = true;
            }
        }
        bool is_raw_data_released() const override { return raw_data_released_; }

        void release_host_weight_data() override
        {
            unpinHostMemory(); // Must unpin BEFORE freeing — CUDA/HIP tracks pinned regions
            release_raw_data();
            {
                decltype(dequant_cache_) tmp;
                std::swap(dequant_cache_, tmp);
            }
            mmap_owner_.reset();
        }

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

        // IINT8Unpackable interface
        __attribute__((always_inline)) void unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const override
        {
            const size_t super_blocks_per_row = (shape_[1] + IQ1_MBlock::BLOCK_SIZE - 1) / IQ1_MBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ1_MBlock *blocks = reinterpret_cast<const IQ1_MBlock *>(data_ptr);

            const size_t super_block_idx = (row_idx * super_blocks_per_row) + (k_block_offset / 8);
            const size_t sub_block_idx = k_block_offset % 8;

            simd::unpack_iq1_m_to_int8(blocks[super_block_idx], sub_block_idx, output);
        }

        __attribute__((always_inline)) float get_block_scale(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t super_blocks_per_row = (shape_[1] + IQ1_MBlock::BLOCK_SIZE - 1) / IQ1_MBlock::BLOCK_SIZE;
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ1_MBlock *blocks = reinterpret_cast<const IQ1_MBlock *>(data_ptr);

            const size_t super_block_idx = (row_idx * super_blocks_per_row) + (k_block_offset / 8);
            const size_t sub_block_idx = k_block_offset % 8;

            return simd::get_iq1_m_scale(blocks[super_block_idx], sub_block_idx);
        }

        size_t superblock_size() const override { return 256; }
        const NativeVnniFormatInfo *vnniFormatInfo() const override
        {
            static constexpr NativeVnniFormatInfo info{17, 6, true, true, false, 1.125f};
            return &info;
        }
        void packVnniBlock(const VnniPackContext &ctx, int n, int b) const override;
        void unpack_superblock_to_int8(size_t row_idx, size_t superblock_idx, int8_t *output, float *scales = nullptr, float *mins = nullptr) const override;

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

    protected:
        // ===== Lazy Transfer Accessors (Phase 3) =====
        void *raw_host_data_ptr() override
        {
            return is_view_ ? const_cast<uint8_t *>(raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        const void *raw_host_data_ptr() const override
        {
            return is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        }
        size_t byte_size() const override { return data_byte_size_ > 0 ? data_byte_size_ : raw_data_.size(); }

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
        std::shared_ptr<void> mmap_owner_;   // Keeps mmap region alive for zero-copy tensors
        size_t data_byte_size_ = 0;          // Byte size for mmap-backed tensors (raw_data_ is empty)

        DeviceId device_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        bool raw_data_released_ = false; // Track if raw data was released after GEMM pack
    };

} // namespace llaminar2
