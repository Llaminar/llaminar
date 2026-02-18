/**
 * @file QuantisedGemmKernel.h
 * @brief High-performance quantized GEMM kernel with JIT code generation
 * @author David Sanftenberg
 *
 * @details
 * This file implements a production-grade quantized matrix multiplication kernel
 * that leverages AVX-512 VNNI instructions through JIT-compiled code. The kernel
 * orchestrates weight packing, activation quantization, and dispatches to either
 * M=1 or M=2 JIT kernels based on sequence length.
 *
 * ## Architecture Overview
 *
 * ```
 *                          QuantisedGemmKernel
 *                                  │
 *              ┌───────────────────┼───────────────────┐
 *              │                   │                   │
 *       Weight Packing      A Quantization       JIT Dispatch
 *              │                   │                   │
 *       IINT8Unpackable     FP32 → Q8_1      M1 or M2 Kernel
 *              │                   │                   │
 *       QuantisedPackedWeights   Q8_1Block[]   AVX-512 VNNI
 * ```
 *
 * ## Quantization Format
 *
 * **Activation Quantization (Q8_1 per-block):**
 * - Block size: 32 elements
 * - Scale: FP16 (computed as max_abs / 127)
 * - Sum: INT16 (sum of quantized values for asymmetric correction)
 * - Values: INT8 (quantized activations)
 *
 * **Weight Packing Layout:**
 * - Weights are repacked from native format to VNNI-optimal layout
 * - Layout: `[N/64][K/4][64][4]` for vectorized loads
 * - Auxiliary arrays: scales, compensation (column sums), mins (for IQ4_NL)
 *
 * ## Kernel Selection
 *
 * | Sequence Length | Kernel | Strategy |
 * |-----------------|--------|----------|
 * | m = 1           | M1     | Single-row, maximize ILP |
 * | m >= 2          | M2     | Two-row, better throughput |
 *
 * ## Cache-Aware Blocking
 *
 * The kernel implements cache-aware N-dimension blocking:
 * - L2 constraint: Block B data should fit in L2 cache
 * - L3 constraint: All thread blocks should fit in shared L3
 * - Block size is dynamically computed based on detected cache sizes
 *
 * ## Parallelization Strategy
 *
 * - **Quantization**: Parallelized over M (rows) or M×K for small sequences
 * - **GEMM**: Parallelized over N-blocks for B-stationary caching
 * - Uses OpenMP with dynamic scheduling for load balancing
 *
 * ## Fused Operations
 *
 * The kernel supports fusing the following operations into a single pass:
 * - Bias addition
 * - Attention mask addition
 * - Softmax (first pass: local max and exp-sum)
 * - SwiGLU activation (for FFN blocks)
 *
 * ## Usage Example
 *
 * ```cpp
 * // Create kernel from quantized weights (any IINT8Unpackable type)
 * auto kernel = std::make_unique<QuantisedGemmKernel>(weights_tensor.get());
 *
 * // Basic GEMM: C = A @ W
 * kernel->multiply(A_data, C_data, m, n, k, false, 1.0f, 0.0f, nullptr, -1);
 *
 * // Fused GEMM with bias and softmax
 * kernel->multiply_fused(A_data, C_data, m, n, k,
 *                        bias, mask, true, local_max, local_sum,
 *                        false, 1.0f, 0.0f, nullptr, -1);
 * ```
 *
 * ## Performance Considerations
 *
 * - Weight packing is done once at kernel construction (amortized cost)
 * - Activation quantization happens per-GEMM call (necessary for dynamic values)
 * - JIT compilation happens once per kernel type (static instances)
 * - Prefetching is used in inner loops for B matrix data
 *
 * @see QuantisedGemmJit_M1 Single-row JIT kernel implementation
 * @see QuantisedGemmJit_M2 Two-row JIT kernel implementation
 * @see IINT8Unpackable Interface for quantized tensor unpacking
 */

#pragma once

#include <immintrin.h>
#include <atomic>
#include "QuantisedGemmJit_M1.h"
#include "QuantisedGemmJit_M2.h"
#include "QuantisedGemmJit_Q8_1_OnlineSoftmax.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/FP16Utils.h"
#include "../../../tensors/SIMDHelpers.h"
#include "../../../tensors/QuantizationUtils.h"
#include "../../../utils/CPUFeatures.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/KernelProfiler.h"
#include "../../../utils/Logger.h"
#include "../../../utils/OpenMPUtils.h"
#include <vector>
#include <memory>
#include <mutex>
#include <omp.h>
#include <cstring>
#include <iostream>

namespace llaminar2
{
    namespace gemm_v4
    {

        /**
         * @brief High-performance quantized GEMM kernel using AVX-512 VNNI JIT code
         *
         * This class implements the ITensorGemm interface and provides optimized
         * matrix multiplication for quantized weight tensors. Weights are packed
         * at construction time into a VNNI-optimal format, and activations are
         * quantized on-the-fly during each multiply call.
         *
         * @details
         * ## Key Features
         * - Supports any tensor type implementing IINT8Unpackable
         * - Dynamic activation quantization (FP32 → Q8_1)
         * - Cache-aware N-dimension blocking
         * - Fused post-operations (bias, mask, softmax, SwiGLU)
         *
         * ## Thread Safety
         * - Construction is thread-safe (uses OpenMP for parallel packing)
         * - multiply() is thread-safe when called from different threads with
         *   different output buffers
         * - JIT kernel instances are static and thread-safe
         */
        class QuantisedGemmKernel : public ITensorGemm
        {
        public:
            /**
             * @brief Generic constructor for any quantized tensor type
             *
             * The tensor must implement IINT8Unpackable interface. If it doesn't,
             * pack_weights_generic will print an error and the kernel will not work.
             * This allows gradual rollout as more tensor types implement IINT8Unpackable.
             *
             * @param weights Pointer to quantized weight tensor (must implement IINT8Unpackable)
             *
             * @note The weights are copied during construction; the original tensor
             *       can be modified or destroyed after construction.
             */
            QuantisedGemmKernel(const TensorBase *weights)
            {
                pack_weights_generic(weights);
            }

            /**
             * @brief Row-sliced constructor for tensor parallelism weight sharding
             *
             * Creates a kernel that only packs rows [row_start, row_end) from the weight tensor.
             * This is used for row-parallel tensor parallelism where each MPI rank owns
             * a slice of the output dimension.
             *
             * For a weight matrix [N, K]:
             * - Full kernel: packs all N rows
             * - Sliced kernel: packs only (row_end - row_start) rows
             *
             * @param weights Pointer to quantized weight tensor (must implement IINT8Unpackable)
             * @param row_start First row to include (0-indexed)
             * @param row_end One past the last row to include
             *
             * @note The resulting kernel has N = (row_end - row_start), K unchanged
             * @note This enables true row-parallel GEMM without dequantizing weights
             */
            QuantisedGemmKernel(const TensorBase *weights, int row_start, int row_end)
            {
                pack_weights_generic(weights, row_start, row_end);
            }

            /**
             * @brief Legacy constructor for Q8_1 tensors (deprecated)
             * @param weights Pointer to Q8_1 quantized weight tensor
             * @deprecated Use the generic TensorBase constructor instead
             */
            QuantisedGemmKernel(const Q8_1Tensor *weights)
            {
                pack_weights_generic(weights);
            }

            /**
             * @brief Legacy constructor for Q8_0 tensors (deprecated)
             * @param weights Pointer to Q8_0 quantized weight tensor
             * @deprecated Use the generic TensorBase constructor instead
             */
            QuantisedGemmKernel(const Q8_0Tensor *weights)
            {
                pack_weights_generic(weights);
            }

            /**
             * @brief Legacy constructor for Q4_0 tensors (deprecated)
             * @param weights Pointer to Q4_0 quantized weight tensor
             * @deprecated Use the generic TensorBase constructor instead
             */
            QuantisedGemmKernel(const Q4_0Tensor *weights)
            {
                pack_weights_generic(weights);
            }

            /**
             * @brief Constructor that references externally-owned packed weights
             *
             * This constructor does NOT pack weights - it expects pre-packed weights
             * that are owned elsewhere (typically in the tensor's cache_). This enables
             * the pack-once-use-many pattern where packed data outlives individual kernels.
             *
             * @param packed Pointer to pre-packed weights (must outlive this kernel)
             * @note Use this when tensor owns packed data and you want kernel to reference it
             */
            explicit QuantisedGemmKernel(const QuantisedPackedWeights *packed)
                : external_packed_(packed)
            {
                // No packing - just reference external data
            }

            /**
             * @brief Access packed weight dimensions
             * @return Reference to packed weights structure (N, K, data)
             */
            const QuantisedPackedWeights &packed_weights() const
            {
                return external_packed_ ? *external_packed_ : packed_weights_;
            }

            /**
             * @brief Get the output dimension (N) of the packed weights
             * @return Number of output features this kernel produces
             *
             * This is useful for callers to determine weight dimensions when
             * weights may have been sharded (tensor parallelism).
             */
            int get_n() const { return packed_weights().N; }

            /**
             * @brief Get the input dimension (K) of the packed weights
             * @return Number of input features this kernel expects
             *
             * This is useful for callers to determine weight dimensions when
             * weights may have been sharded (tensor parallelism).
             */
            int get_k() const { return packed_weights().K; }

            // =========================================================================
            // Static Packing API - for tensor-owned packed weights
            // =========================================================================

            /**
             * @brief Pack weights into an externally-owned QuantisedPackedWeights
             *
             * This static method packs quantized weights into the VNNI-optimal format
             * without creating a kernel. Use this when you want the tensor to own
             * the packed data and have multiple kernels reference it.
             *
             * @param weights Source quantized tensor (must implement IINT8Unpackable)
             * @param out_packed Output structure to populate with packed data
             * @param row_start First row to pack (default 0)
             * @param row_end Last row to pack, -1 = all rows (default -1)
             * @return true on success, false on error
             *
             * @note See inline implementation below in public section.
             */

        private:
            /**
             * @brief Pack a single 32-element block into VNNI-optimal layout
             *
             * This helper function packs one quantization block (32 INT8 values)
             * into the strided layout expected by the JIT kernels.
             *
             * @param n Row index in the weight matrix
             * @param k_blk K-block index (each block covers 32 K elements)
             * @param row_base_offset Base byte offset for this row in packed_data
             * @param temp_vals 32 INT8 values to pack
             * @param scale Block scale factor (FP32)
             * @param min_val Block minimum value for asymmetric correction (FP32)
             * @param N_padded Padded N dimension (multiple of 64)
             *
             * @details
             * The packed layout is `[N/64][K/4][64][4]`:
             * - Groups of 64 columns are processed together
             * - Within each group, K is processed in steps of 4 (for vpdpbusd)
             * - The innermost dimension holds 4 consecutive K values
             *
             * Additionally stores:
             * - compensation: Sum of quantized values (for INT8→UINT8 correction)
             * - scales: Block scale factor
             * - mins: Minimum value for IQ4_NL asymmetric correction
             */
            __attribute__((always_inline)) void pack_single_block(
                int n,
                int k_blk,
                size_t row_base_offset,
                const int8_t *temp_vals,
                float scale,
                float min_val,
                int N_padded)
            {
                // Compute compensation: sum of all quantized values in block
                // This is used to correct for the INT8→UINT8 conversion in JIT
                int32_t sum = 0;
#pragma omp simd reduction(+ : sum)
                for (int i = 0; i < 32; ++i)
                {
                    sum += temp_vals[i];
                }

                // Calculate destination offset in packed layout
                // k_blk corresponds to 32 K elements
                // Each K step of 4 uses 256 bytes (64 columns × 4 bytes)
                // So k_blk maps to k_blk * 8 groups of 4
                size_t block_offset = row_base_offset + (size_t)(k_blk * 8) * 256;

                int8_t *dst_ptr = packed_weights_.packed_data.data() + block_offset;

                // Pack 32 bytes into 8 groups of 4, with stride 256 between groups
                // This matches the access pattern in the JIT inner loop
                for (int j = 0; j < 8; ++j)
                {
                    std::memcpy(dst_ptr + j * 256, &temp_vals[j * 4], 4);
                }

                // Store auxiliary data for the JIT kernel
                packed_weights_.compensation[k_blk * N_padded + n] = sum;
                packed_weights_.scales[k_blk * N_padded + n] = scale;
                packed_weights_.mins[k_blk * N_padded + n] = min_val;
            }

            /**
             * @brief Pack weights from any quantized tensor into VNNI-optimal layout
             *
             * This method converts weights from their native storage format into a
             * layout optimized for AVX-512 VNNI computation. The tensor must implement
             * the IINT8Unpackable interface.
             *
             * @param weights Quantized weight tensor to pack
             * @param row_start First row to pack (default: 0)
             * @param row_end One past last row to pack (default: -1 means all rows)
             *
             * @details
             * ## Packing Process
             *
             * 1. **Dimension Setup**: Extract N (output features) and K (input features)
             * 2. **Buffer Allocation**: Allocate packed_data, compensation, scales, mins
             * 3. **Block Iteration**: Process 32-element K blocks for each output row
             * 4. **Layout Transformation**: Repack into `[N/64][K/4][64][4]` format
             *
             * ## Row Slicing for Tensor Parallelism
             *
             * When row_start and row_end are specified, only rows [row_start, row_end)
             * are packed. The resulting kernel has N = (row_end - row_start).
             * The IINT8Unpackable interface is called with the original tensor's row
             * indices, so the quantized data is read correctly.
             *
             * ## Superblock Optimization
             *
             * For tensor formats with 256-element superblocks (like Q6_K), we process
             * 8 blocks at a time using unpack_superblock_to_int8() for better cache
             * utilization and fewer function calls.
             *
             * ## Thread Parallelization
             *
             * Packing is parallelized over N (output rows) using OpenMP, which provides
             * good load balancing and cache locality since each thread works on
             * independent rows.
             *
             * @note This method is called from the constructor. Weight packing is
             *       typically the slowest part of kernel initialization, but it only
             *       happens once per weight matrix.
             */
            void pack_weights_generic(const TensorBase *weights, int row_start = 0, int row_end = -1)
            {
                // Delegate to static method that packs into our member
                if (!packWeightsInto(weights, packed_weights_, row_start, row_end))
                {
                    LOG_ERROR("[QuantisedGemmKernel] Failed to pack weights");
                }
            }

        public:
            /**
             * @brief Static method to pack weights into an external QuantisedPackedWeights
             *
             * This enables tensor-owned packed weights where the tensor stores the packed
             * data in its cache_ and multiple kernels can reference it.
             */
            static bool packWeightsInto(const TensorBase *weights, QuantisedPackedWeights &out, int row_start = 0, int row_end = -1)
            {
                // Weights are typically [N, K] (out_features, in_features)
                int full_N = weights->shape()[0];
                int K = weights->shape()[1];

                // Handle row slicing for tensor parallelism
                if (row_end < 0)
                    row_end = full_N; // Default: all rows
                if (row_start < 0)
                    row_start = 0;
                if (row_end > full_N)
                    row_end = full_N;
                if (row_start >= row_end)
                {
                    LOG_ERROR("[QuantisedGemmKernel] Invalid row range: [" << row_start << ", " << row_end << ")");
                    return false;
                }

                int N = row_end - row_start; // Number of rows to pack

                out.K = K;
                out.N = N;

                // Pad N to multiple of 64 for SIMD blocking
                int N_padded = (N + 63) / 64 * 64;
                // K_blocks covers all K elements (32 per block)
                int K_blocks = (K + 31) / 32;
                // K_padded is K rounded up to multiple of 32 (VNNI block size)
                int K_padded = K_blocks * 32;

                // Allocate packed weight buffers
                // Layout: [N/64][K_padded/4][64][4] flattened, so total size = K_padded * N_padded
                out.packed_data.resize(K_padded * N_padded);
                out.compensation.resize(K_blocks * N_padded);
                out.scales.resize(K_blocks * N_padded);
                out.mins.resize(K_blocks * N_padded);

                // Track if we have any non-zero mins (for JIT optimization)
                std::atomic<bool> has_mins_atomic(false);

                // Verify tensor implements IINT8Unpackable interface
                const IINT8Unpackable *unpackable = dynamic_cast<const IINT8Unpackable *>(weights);
                if (!unpackable)
                {
                    LOG_ERROR("Tensor type " << (int)weights->native_type() << " does not implement IINT8Unpackable");
                    return false;
                }

                // Check if tensor uses 256-element superblocks (e.g., K-quants)
                bool use_superblock = (unpackable->superblock_size() == 256);

                // Parallel pack over N rows
                // Note: 'n' is the local row index (0 to N-1) in the packed output
                //       'src_row' = row_start + n is the row index in the source tensor
#pragma omp parallel for schedule(static)
                for (int n = 0; n < N; ++n)
                {
                    int src_row = row_start + n; // Map local row to source tensor row

                    // Calculate base offset for this row in packed layout
                    // Layout: [N/64][K/4][64][4]
                    int n_blk = n / 64;
                    int n_rem = n % 64;
                    size_t row_base_offset = (size_t)n_blk * (K * 64) + n_rem * 4;

                    int k_blk = 0;

                    // Process superblocks first (8 blocks at a time) if supported
                    if (use_superblock)
                    {
                        int K_superblocks = K_blocks / 8;
                        for (int k_sb = 0; k_sb < K_superblocks; ++k_sb)
                        {
                            // Unpack 256 elements (8 × 32-element blocks)
                            int8_t sb_vals[256];
                            float sb_scales[8];
                            float sb_mins[8];

                            // Use src_row to read from the original tensor's row
                            unpackable->unpack_superblock_to_int8(src_row, k_sb, sb_vals, sb_scales, sb_mins);

                            // Pack each of the 8 blocks (using local row index 'n' for output)
                            for (int i = 0; i < 8; ++i)
                            {
                                if (sb_mins[i] != 0.0f)
                                    has_mins_atomic.store(true, std::memory_order_relaxed);
                                pack_single_block_static(out, n, k_blk + i, row_base_offset, sb_vals + i * 32, sb_scales[i], sb_mins[i], N_padded);
                            }
                            k_blk += 8;
                        }
                    }

                    // Process remaining blocks individually
                    for (; k_blk < K_blocks; ++k_blk)
                    {
                        int8_t temp_vals[32];
                        // Use src_row to read from the original tensor's row
                        unpackable->unpack_block_to_int8(src_row, k_blk, temp_vals);
                        float scale = unpackable->get_block_scale(src_row, k_blk);
                        float min_val = unpackable->get_block_min(src_row, k_blk);
                        if (min_val != 0.0f)
                            has_mins_atomic.store(true, std::memory_order_relaxed);

                        pack_single_block_static(out, n, k_blk, row_base_offset, temp_vals, scale, min_val, N_padded);
                    }
                }
                out.has_mins = has_mins_atomic.load();
                return true;
            }

        private:
            /**
             * @brief Static version of pack_single_block for use in packWeightsInto
             */
            __attribute__((always_inline)) static void pack_single_block_static(
                QuantisedPackedWeights &out,
                int n,
                int k_blk,
                size_t row_base_offset,
                const int8_t *temp_vals,
                float scale,
                float min_val,
                int N_padded)
            {
                int K = out.K;

                // Compute compensation (sum of INT8 values) for INT8→UINT8 correction
                int comp = 0;
                for (int i = 0; i < 32; ++i)
                {
                    comp += temp_vals[i];
                }

                // Store metadata at block position
                out.compensation[k_blk * N_padded + n] = comp;
                out.scales[k_blk * N_padded + n] = scale;
                out.mins[k_blk * N_padded + n] = min_val;

                // Pack into VNNI layout: [N/64][K/4][64][4]
                for (int i = 0; i < 32; i += 4)
                {
                    size_t pack_idx = row_base_offset + (size_t)(k_blk * 32 + i) * 64;
                    out.packed_data[pack_idx + 0] = temp_vals[i + 0];
                    out.packed_data[pack_idx + 1] = temp_vals[i + 1];
                    out.packed_data[pack_idx + 2] = temp_vals[i + 2];
                    out.packed_data[pack_idx + 3] = temp_vals[i + 3];
                }
            }

            // =========================================================================
            // Fused Multi-GEMM API - Quantize once, execute multiple projections
            // =========================================================================

        public:
            /**
             * @brief Descriptor for a single projection in multiply_fused_multi
             *
             * Each projection specifies a kernel (with packed weights), output buffer,
             * and optional fused operations. All projections share the same quantized
             * input buffer.
             */
            struct FusedProjection
            {
                QuantisedGemmKernel *kernel;       ///< Kernel with packed weights [n, k]
                float *output;                     ///< Output buffer [m, n]
                int n;                             ///< Output dimension (columns)
                const TensorBase *bias = nullptr;  ///< Optional bias tensor [n]
                const float *gate_input = nullptr; ///< Optional gate for SwiGLU [m, n]
                bool do_swiglu = false;            ///< Whether to apply SwiGLU fusion
                const char *name = nullptr;        ///< Name for debug logging
            };

            /**
             * @brief Execute multiple GEMMs sharing a single quantized input buffer
             *
             * This is the optimal API for patterns like Q/K/V projections or gate/up
             * FFN projections, where the same input is projected through multiple
             * weight matrices. The input is quantized once (FP32 → Q8_1), then all
             * projections execute against the shared buffer.
             *
             * @param input FP32 input activations [m, k]
             * @param projections Vector of projection descriptors
             * @param m Number of rows (batch_size * seq_len)
             * @param k Input dimension (must match all kernels' K dimension)
             * @return true on success, false on error
             *
             * @details
             * ## Computation Flow
             *
             * ```
             * [FP32 Input] ──► Quantize ──► [Q8_1 Buffer]
             *                                    │
             *                    ┌───────────────┼───────────────┐
             *                    ↓               ↓               ↓
             *              GEMM(kernel_0)  GEMM(kernel_1)  GEMM(kernel_2)
             *                    ↓               ↓               ↓
             *              [Output 0]      [Output 1]      [Output 2]
             * ```
             *
             * ## Performance Benefits
             * - Saves (N-1) quantization passes vs calling multiply() N times
             * - Input stays in cache across all projections
             * - No intermediate buffer management by caller
             *
             * ## Example Usage
             *
             * ```cpp
             * auto* prepared_q = KernelFactory::getOrCreatePreparedGemmWeights(wq, DeviceId::cpu());
             * auto* prepared_k = KernelFactory::getOrCreatePreparedGemmWeights(wk, DeviceId::cpu());
             * auto* prepared_v = KernelFactory::getOrCreatePreparedGemmWeights(wv, DeviceId::cpu());
             * auto* kernel_q = static_cast<QuantisedGemmKernel*>(KernelFactory::getOrCreateGemmEngine(prepared_q));
             * auto* kernel_k = static_cast<QuantisedGemmKernel*>(KernelFactory::getOrCreateGemmEngine(prepared_k));
             * auto* kernel_v = static_cast<QuantisedGemmKernel*>(KernelFactory::getOrCreateGemmEngine(prepared_v));
             *
             * QuantisedGemmKernel::multiply_fused_multi(
             *     normalized->data(),
             *     {
             *         {kernel_q, Q->mutable_data(), n_heads * head_dim, nullptr, nullptr, false, "Q"},
             *         {kernel_k, K->mutable_data(), n_kv_heads * head_dim, nullptr, nullptr, false, "K"},
             *         {kernel_v, V->mutable_data(), n_kv_heads * head_dim, nullptr, nullptr, false, "V"}
             *     },
             *     seq_len, d_model
             * );
             * ```
             *
             * @note All kernels must have the same K dimension
             * @note Thread-safe: can be called concurrently with different inputs
             */
            static bool multiply_fused_multi(
                const float *input,
                const std::vector<FusedProjection> &projections,
                int m, int k)
            {
                if (!input || projections.empty())
                {
                    LOG_ERROR("[multiply_fused_multi] Null input or empty projections");
                    return false;
                }

                if (m <= 0 || k <= 0)
                {
                    LOG_ERROR("[multiply_fused_multi] Invalid dimensions: m=" << m << " k=" << k);
                    return false;
                }

                // Validate all projections
                for (size_t i = 0; i < projections.size(); ++i)
                {
                    const auto &proj = projections[i];
                    if (!proj.kernel)
                    {
                        LOG_ERROR("[multiply_fused_multi] Projection " << i << " has null kernel");
                        return false;
                    }
                    if (!proj.output)
                    {
                        LOG_ERROR("[multiply_fused_multi] Projection " << i << " has null output");
                        return false;
                    }
                    if (proj.n <= 0)
                    {
                        LOG_ERROR("[multiply_fused_multi] Projection " << i << " has invalid n=" << proj.n);
                        return false;
                    }
                    // Verify K dimension matches
                    if (proj.kernel->get_k() != k)
                    {
                        LOG_ERROR("[multiply_fused_multi] Projection " << i
                                                                       << " kernel K=" << proj.kernel->get_k() << " doesn't match input k=" << k);
                        return false;
                    }
                    // Verify N dimension matches
                    if (proj.kernel->get_n() != proj.n)
                    {
                        LOG_ERROR("[multiply_fused_multi] Projection " << i
                                                                       << " n=" << proj.n << " doesn't match kernel N=" << proj.kernel->get_n());
                        return false;
                    }
                }

                // Step 1: Allocate shared Q8_1 buffer for quantized activations
                int k_blocks = (k + 31) / 32;
                size_t buffer_size = static_cast<size_t>(m) * k_blocks * sizeof(Q8_1Block);
                std::vector<uint8_t> q8_1_buffer(buffer_size);

                // Debug: Dump first 32 floats to see what's being quantized
                if (m >= 1 && k >= 32)
                {
                    LOG_TRACE("[multiply_fused_multi] input[0:8]="
                              << std::setprecision(6) << input[0] << "," << input[1] << "," << input[2] << "," << input[3]
                              << "," << input[4] << "," << input[5] << "," << input[6] << "," << input[7]);
                }

                // Step 2: Quantize activations once
                bool success;
                {
                    KERNEL_PROFILE_SCOPE(KernelType::QUANTIZE_Q8);
                    success = projections[0].kernel->quantize_activations(input, q8_1_buffer.data(), m, k);
                }
                if (!success)
                {
                    LOG_ERROR("[multiply_fused_multi] Failed to quantize activations");
                    return false;
                }

                // Debug: dump first Q8_1 block for comparison
                const Q8_1Block *blocks = reinterpret_cast<const Q8_1Block *>(q8_1_buffer.data());
                LOG_DEBUG("[multiply_fused_multi] First Q8_1 block: d=" << blocks[0].d
                                                                        << " sum_qs=" << blocks[0].sum_qs
                                                                        << " qs[0:4]=" << (int)blocks[0].qs[0] << ","
                                                                        << (int)blocks[0].qs[1] << ","
                                                                        << (int)blocks[0].qs[2] << ","
                                                                        << (int)blocks[0].qs[3]);

                // Step 3: Execute all projections with shared quantized activations
                for (size_t i = 0; i < projections.size(); ++i)
                {
                    const auto &proj = projections[i];
                    const char *name = proj.name ? proj.name : "projection";

                    // Debug: Log kernel and packed weight pointer for Q projection tracing
                    if (i == 0 && proj.n == 896 && k == 896)
                    {
                        const auto &pw = proj.kernel->packed_weights();
                        LOG_TRACE("[multiply_fused_multi] " << name << " kernel=" << static_cast<const void *>(proj.kernel)
                                                            << " pw.packed_data.data()=" << static_cast<const void *>(pw.packed_data.data())
                                                            << " pw.N=" << pw.N << " pw.K=" << pw.K);
                    }

                    // DEBUG: Log output buffer BEFORE GEMM to check for stale data
                    if (i == 0 && proj.n == 896 && k == 896)
                    {
                        LOG_TRACE("[multiply_fused_multi] " << proj.name << " BEFORE: output=" << static_cast<const void *>(proj.output)
                                                            << " output[0:4]=" << std::setprecision(10)
                                                            << proj.output[0] << "," << proj.output[1] << ","
                                                            << proj.output[2] << "," << proj.output[3]);
                    }

                    // Build fused ops configuration
                    GemmFusedOps fused_ops = proj.do_swiglu
                                                 ? GemmFusedOps::swiglu(proj.gate_input)
                                                 : GemmFusedOps::none();

                    success = proj.kernel->multiply_with_precomputed_q8_1(
                        q8_1_buffer.data(),
                        proj.output,
                        m, proj.n, k,
                        proj.bias,
                        false,       // No accumulation
                        1.0f, 0.0f,  // alpha=1, beta=0
                        nullptr, -1, // No MPI context, CPU device
                        fused_ops);

                    if (!success)
                    {
                        LOG_ERROR("[multiply_fused_multi] " << name << " GEMM failed");
                        return false;
                    }

                    // Debug: dump first 4 output values
                    LOG_DEBUG("[multiply_fused_multi] " << name << " output[0:4]="
                                                        << std::setprecision(10) << proj.output[0] << ","
                                                        << proj.output[1] << "," << proj.output[2] << "," << proj.output[3]);
                }

                return true;
            }

        public:
            /**
             * @brief Check if this kernel supports the specified device
             *
             * @param device_idx Device index (-1 = CPU, ≥0 = GPU)
             * @return true if device is supported, false otherwise
             *
             * @note This kernel only supports CPU execution (device_idx == -1)
             */
            bool supports_device(int device_idx) const override
            {
                return device_idx == -1;
            }

            /**
             * @brief Check if this kernel supports optimized fused multi-projection
             *
             * QuantisedGemmKernel supports optimized fusion: input is quantized once
             * and all projections share the Q8_1 buffer.
             *
             * @return true - this kernel has optimized fusion via multiply_fused_multi
             */
            bool supports_fused_projection() const override
            {
                return true;
            }

            /**
             * @brief Fused multi-projection GEMM using shared Q8_1 activation buffer
             *
             * This override provides optimized fusion for quantized GEMM:
             * - Input is quantized to Q8_1 once
             * - All projections execute against the shared quantized buffer
             * - Saves (N-1) quantization passes vs calling multiply() N times
             *
             * @param input FP32 input activations [m, k]
             * @param projections Vector of projection descriptors
             * @param m Number of rows
             * @param k Input dimension
             * @param mpi_ctx MPI context (unused)
             * @param device_idx Device index (-1 = CPU)
             *
             * @return true on success, false on error
             */
            bool multiply_fused(
                const float *input,
                const std::vector<FusedProjectionDesc> &projections,
                int m, int k,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                DeviceWorkspaceManager *workspace = nullptr) override
            {
                (void)mpi_ctx;
                (void)device_idx;
                (void)workspace; // CPU kernel doesn't need external workspace

                // Convert ITensorGemm::FusedProjectionDesc to internal FusedProjection format
                std::vector<FusedProjection> internal_projections;
                internal_projections.reserve(projections.size());

                for (const auto &proj : projections)
                {
                    // The kernel in FusedProjectionDesc must be a QuantisedGemmKernel
                    auto *qkernel = dynamic_cast<QuantisedGemmKernel *>(proj.kernel);
                    if (!qkernel)
                    {
                        LOG_ERROR("[QuantisedGemmKernel::multiply_fused] Projection kernel is not QuantisedGemmKernel");
                        return false;
                    }
                    internal_projections.push_back({qkernel, proj.output, proj.n, proj.bias, proj.gate_input, proj.do_swiglu, proj.name});
                }

                // Delegate to the optimized static method
                return multiply_fused_multi(input, internal_projections, m, k);
            }

            /**
             * @brief Matrix multiply with packed quantized weights: C = alpha * A @ B + beta * C
             *
             * @param A Input activations [m, k]
             * @param C Output [m, n], will be overwritten or accumulated based on beta
             * @param m Batch size (sequence length)
             * @param n Output features (must match packed_weights_.N)
             * @param k Input features (must match packed_weights_.K)
             * @param transpose_B Ignored - weights are pre-transposed during packing
             * @param alpha Scale factor for A @ B
             * @param beta If 0, overwrite C. If non-zero, accumulate: C = alpha*A@B + beta*C
             * @param ctx MPI context (unused for local compute)
             * @param device_idx Device index (-1 = CPU)
             *
             * @return true on success, false on dimension mismatch
             */
            bool multiply(const float *A, float *C, int m, int n, int k, bool transpose_B, float alpha, float beta, const MPIContext *ctx, int device_idx, DeviceWorkspaceManager *workspace = nullptr) override
            {
                // Note: transpose_B is ignored - quantized weights are always pre-transposed during packing
                // Accumulate is determined by beta: beta > 0 means add to existing C
                (void)transpose_B;
                (void)workspace; // CPU kernel doesn't need external workspace
                bool accumulate = (beta != 0.0f);
                return multiply_fused(A, C, m, n, k, nullptr, nullptr, false, nullptr, nullptr, accumulate, alpha, beta, ctx, device_idx);
            }

            // IKernelSnapshotCapable interface
            KernelSnapshotInfo getKernelSnapshotInfo() const override
            {
                return KernelSnapshotInfo::gemm()
                    .withInput("A", "input activations [m, k]", KernelBufferDtype::FP32)
                    .withWeight("B", "quantized weight matrix [n, k] (packed VNNI format)", KernelBufferDtype::INT8)
                    .withOutput("C", "output matrix [m, n]", KernelBufferDtype::FP32)
                    .withScalar("m", "batch dimension", KernelBufferDtype::INT32)
                    .withScalar("n", "output features", KernelBufferDtype::INT32)
                    .withScalar("k", "input features", KernelBufferDtype::INT32)
                    .withScalar("alpha", "output scale factor")
                    .withScalar("beta", "accumulate scale factor");
            }

            /**
             * @brief Fused GEMM with optional post-operations (bias, mask, softmax, SwiGLU)
             *
             * Performs quantized matrix multiplication with optional fused operations
             * in a single pass through the data. This is the main workhorse method
             * that implements the full computation pipeline.
             *
             * @param A Input activations [m, k] in FP32 (will be quantized internally)
             * @param C Output matrix [m, n] in FP32
             * @param m Batch size / sequence length
             * @param n Output features (columns)
             * @param k Input features (rows in weight matrix)
             * @param bias Optional bias vector [n], added to each row of C
             * @param mask Optional attention mask [m, n], added to C (use -inf for masking)
             * @param do_softmax If true, compute local max and exp-sum for online softmax
             * @param local_max Output buffer for per-row max values (required if do_softmax)
             * @param local_sum Output buffer for per-row sum(exp) values (required if do_softmax)
             * @param accumulate If true, add to existing C; if false, overwrite C
             * @param alpha Scale factor for matrix product (typically 1.0)
             * @param beta Scale factor for existing C (0.0 to overwrite, 1.0 to accumulate)
             * @param ctx MPI context (unused, for interface compatibility)
             * @param device_idx Device index (-1 for CPU)
             * @param gate_input Optional gate values for SwiGLU [m, n]
             * @param do_swiglu If true, apply SwiGLU: output *= swish(gate_input)
             *
             * @return true on success, false on error
             *
             * @details
             * ## Computation Pipeline
             *
             * 1. **C Initialization**: Zero or scale C based on beta
             * 2. **A Quantization**: FP32 → Q8_1 (parallel over M or M×K)
             * 3. **GEMM**: Dispatch to M1 or M2 JIT kernel (parallel over N-blocks)
             * 4. **Post-ops**: Applied in JIT kernel (bias, mask, softmax, SwiGLU)
             *
             * ## Softmax Notes
             *
             * When do_softmax=true, only the first pass of online softmax is computed:
             * - local_max[row] = max(C[row, :])
             * - local_sum[row] = sum(exp(C[row, :] - local_max[row]))
             *
             * The caller must complete softmax by computing:
             * - global_max = max(local_max across tiles)
             * - Rescale and normalize in a second pass
             *
             * ## SwiGLU Notes
             *
             * When do_swiglu=true, the output is element-wise multiplied by swish(gate):
             * - output[i] = C[i] * (gate[i] * sigmoid(gate[i]))
             *
             * This is used in FFN blocks where up_proj is computed through this GEMM
             * and gate_proj is passed as gate_input.
             */
            bool multiply_fused(const float *A, float *C, int m, int n, int k,
                                const float *bias, const float *mask, bool do_softmax,
                                float *local_max, float *local_sum,
                                bool accumulate, float alpha, float beta, const MPIContext *ctx, int device_idx,
                                const float *gate_input = nullptr, bool do_swiglu = false)
            {
                const KernelType profile_type = (m == 1) ? KernelType::GEMV_Q8 : KernelType::GEMM_Q8;
                KERNEL_PROFILE_SCOPE(profile_type);

                // Get reference to packed weights (either external or owned)
                const auto &pw = packed_weights();

                // Validate dimensions match packed weights
                if (n != pw.N || k != pw.K)
                {
                    LOG_ERROR("Dimension mismatch in QuantisedGemmKernel: expected N=" << pw.N << ", K=" << pw.K
                                                                                       << " but got N=" << n << ", K=" << k);
                    return false;
                }

                // Get static JIT kernel instances (compiled once, reused)
                static QuantisedGemmJit_M1 jit;    // Single-row kernel
                static QuantisedGemmJit_M2 jit_m2; // Two-row kernel

                auto kernel = jit.get_kernel();
                auto kernel_m2 = jit_m2.get_kernel();

                // Compute block counts for K and N dimensions
                int k_blocks = (k + 31) / 32;       // Number of 32-element K blocks
                int blocks_per_row = (n + 63) / 64; // Number of 64-element N blocks

                // N must be padded to multiple of 64 (matches packing)
                int N_padded = (n + 63) / 64 * 64;
                bool is_padded = (n != N_padded);

                // Determine how to handle existing C values
                bool needs_zero = (!accumulate || beta == 0.0f);  // Overwrite C
                bool needs_scale = (!needs_zero && beta != 1.0f); // Scale C before accumulate

                // Allocate buffer for quantized activations (all rows, all K blocks)
                // Add 64 bytes padding for alignment
                std::vector<uint8_t> shared_quantized_a(m * k_blocks * sizeof(Q8_1Block) + 64);
                Q8_1Block *all_blocks = reinterpret_cast<Q8_1Block *>(shared_quantized_a.data());

                // =================================================================
                // PARALLEL REGION: C init + A quantization + GEMM
                // =================================================================
                // All three phases run in the same parallel region to minimize
                // OpenMP overhead (single parallel region entry).
                //
                // Layer-fusion support: OMP_WORKSHARE_REGION handles nested parallelism
                // automatically - if already in a parallel region, uses worksharing only.

                auto do_multiply_work = [&]()
                {
                    // ---------------------------------------------------------
                    // Phase 1: Initialize/scale output matrix C
                    // ---------------------------------------------------------
                    // Fused into same parallel region for efficiency
                    if (needs_zero)
                    {
                        // Zero C when not accumulating
#pragma omp for schedule(static) nowait
                        for (int i = 0; i < m; ++i)
                        {
                            std::memset(C + i * n, 0, n * sizeof(float));
                        }
                    }
                    else if (needs_scale)
                    {
#pragma omp for schedule(static) nowait
                        for (int i = 0; i < m; ++i)
                        {
                            float *row = C + i * n;
                            for (int j = 0; j < n; ++j)
                                row[j] *= beta;
                        }
                    }

                    // 1. Quantize A using SIMD-optimized quantization
                    // Uses AVX-512/AVX2 via simd::quantize_single_block()
                    int quant_thresh = debugEnv().gemm.gemm_quant_parallel_threshold;
                    if (quant_thresh == 0)
                        quant_thresh = omp_get_num_threads();

                    const bool k_aligned = (k % 32 == 0);

                    if (m < quant_thresh)
                    {
#pragma omp for collapse(2) schedule(static)
                        for (int i = 0; i < m; ++i)
                        {
                            for (int k_blk = 0; k_blk < k_blocks; ++k_blk)
                            {
                                const float *a_row = A + i * k + k_blk * 32;
                                Q8_1Block *row_blocks = all_blocks + i * k_blocks;

                                int valid_elements = std::min(32, k - k_blk * 32);
                                simd::quantize_single_block(a_row, row_blocks[k_blk], valid_elements);
                            }
                        }
                    }
                    else
                    {
#pragma omp for schedule(static)
                        for (int i = 0; i < m; ++i)
                        {
                            const float *a_row = A + i * k;
                            Q8_1Block *row_blocks = all_blocks + i * k_blocks;

                            if (k_aligned)
                            {
                                // Fast path: K is multiple of 32
                                for (int k_blk = 0; k_blk < k_blocks; ++k_blk)
                                {
                                    simd::quantize_single_block(a_row + k_blk * 32, row_blocks[k_blk], 32);
                                }
                            }
                            else
                            {
                                // General path: handle partial last block
                                for (int k_blk = 0; k_blk < k_blocks - 1; ++k_blk)
                                {
                                    simd::quantize_single_block(a_row + k_blk * 32, row_blocks[k_blk], 32);
                                }
                                if (k_blocks > 0)
                                {
                                    int last_k_blk = k_blocks - 1;
                                    int valid_elements = k - last_k_blk * 32;
                                    simd::quantize_single_block(a_row + last_k_blk * 32, row_blocks[last_k_blk], valid_elements);
                                }
                            }
                        }
                    }

                    // Implicit barrier here

                    // 2. GEMM (Parallel over N blocks and M)
                    // We want to parallelize over N blocks to keep B in L2 cache (B-stationary)
                    // when M is large.

                    // Detect cache sizes
                    int num_threads = omp_get_max_threads(); // Use max threads for planning
                    long long l2_size = cpu_l2_cache_size();
                    if (l2_size == 0)
                        l2_size = 1024 * 1024; // Fallback 1MB

                    long long l3_size = cpu_l3_cache_size();
                    // If L3 is unknown, assume it's large enough to not be the bottleneck compared to L2
                    if (l3_size == 0)
                        l3_size = l2_size * num_threads;

                    // Calculate block size limit
                    // 1. L2 constraint: Block must fit in L2 (use configured % to leave room for A/C)
                    long long l2_limit = (long long)(l2_size * debugEnv().gemm.gemm_l2_limit_pct);

                    // 2. L3 constraint: All threads' blocks must fit in L3
                    // Use configured % of L3 shared among threads
                    long long l3_limit_per_thread = (long long)(l3_size * debugEnv().gemm.gemm_l3_share_pct / num_threads);

                    // Take the tighter constraint
                    long long block_size_limit = std::min(l2_limit, l3_limit_per_thread);

                    // Ensure we have at least some reasonable block size (e.g. 64KB)
                    int min_block_size = debugEnv().gemm.gemm_min_block_size;
                    if (block_size_limit < min_block_size)
                        block_size_limit = min_block_size;

                    int max_n_block = block_size_limit / k;
                    max_n_block = (max_n_block / 64) * 64;
                    if (max_n_block < 64)
                        max_n_block = 64;

                    int target_tasks = num_threads * debugEnv().gemm.gemm_oversubscription_factor;
                    int m_granularity = debugEnv().gemm.gemm_m_task_granularity;
                    int m_tasks = (m + m_granularity - 1) / m_granularity;
                    if (m_tasks < 1)
                        m_tasks = 1;

                    int needed_n_tasks = (target_tasks + m_tasks - 1) / m_tasks;
                    if (needed_n_tasks < 1)
                        needed_n_tasks = 1;

                    int calc_block = (n + needed_n_tasks - 1) / needed_n_tasks;
                    // Align to 64
                    calc_block = (calc_block + 63) / 64 * 64;
                    if (calc_block < 64)
                        calc_block = 64;

                    // Clamp to max_n_block (L2 cache constraint)
                    // Ensure even splitting if we clamp to avoid load imbalance (e.g. 832 vs 64)
                    int n_task_block;
                    if (calc_block > max_n_block)
                    {
                        int num_splits = (n + max_n_block - 1) / max_n_block;
                        int even_block = (n + num_splits - 1) / num_splits;
                        // Align to 64
                        even_block = (even_block + 63) / 64 * 64;
                        n_task_block = even_block;
                    }
                    else
                    {
                        n_task_block = calc_block;
                    }

                    // Check if we need K-tiling (when B-block spills L2 cache)
                    // l2_size is already detected above

                    // Use configured % of L2 cache as threshold for K-tiling
                    // This ensures we tile even for moderate block sizes when M is large
                    const long long L2_CACHE_SIZE = (long long)(l2_size * debugEnv().gemm.gemm_k_tile_threshold_pct);
                    bool needs_k_tiling = ((long long)n_task_block * k > L2_CACHE_SIZE);

                    // Check if we have enough parallelism to avoid collapsing M
                    int num_n_tasks = (n + n_task_block - 1) / n_task_block;
                    bool enough_parallelism = (num_n_tasks >= omp_get_num_threads());

                    if (needs_k_tiling && enough_parallelism)
                    {
                        // K-tiling path: Parallelize N only, tile K inside to reuse B in L2
                        // Calculate dynamic tile size based on L2 cache and n_task_block
                        // We want n_task_block * (k_tile_blocks * 32) * sizeof(int8_t) <= L2_CACHE_SIZE * fraction
                        // Let's use configured % of L2 for B tile
                        long long target_b_size = (long long)(l2_size * debugEnv().gemm.gemm_target_b_size_pct);
                        int k_tile_elements = (int)(target_b_size / n_task_block);
                        int k_tile_blocks = k_tile_elements / 32;

                        // Clamp to reasonable limits
                        if (k_tile_blocks < debugEnv().gemm.gemm_k_tile_min_blocks)
                            k_tile_blocks = debugEnv().gemm.gemm_k_tile_min_blocks;
                        if (k_tile_blocks > debugEnv().gemm.gemm_k_tile_max_blocks)
                            k_tile_blocks = debugEnv().gemm.gemm_k_tile_max_blocks;

#pragma omp for schedule(dynamic)
                        for (int n_task = 0; n_task < n; n_task += n_task_block)
                        {
                            int n_end = std::min(n, n_task + n_task_block);

                            // Iterate over K tiles
                            for (int k_start = 0; k_start < k_blocks; k_start += k_tile_blocks)
                            {
                                int k_count = std::min(k_tile_blocks, k_blocks - k_start);

                                // Iterate over M (reuse B-tile for all M)
                                int unroll = debugEnv().gemm.gemm_m_unroll_factor;
                                if (unroll != 1 && unroll != 2)
                                    unroll = 2;

                                for (int i = 0; i < m; i += unroll)
                                {
                                    int rows_left = m - i;
                                    int rows_to_process = (rows_left >= unroll) ? unroll : rows_left;
                                    if (is_padded && rows_to_process > 1)
                                        rows_to_process = 1;

                                    Q8_1Block *blocks = all_blocks + i * k_blocks + k_start;

                                    for (int n_blk = n_task; n_blk < n_end; n_blk += 64)
                                    {
                                        // Calculate packed weights offset
                                        // Base offset for N-block + Offset for K-tile
                                        // Note: pw is [N_blocks][K_rows][64]
                                        // So offset = (n_blk/64) * (K*64) + (k_start*32)*64
                                        size_t weights_offset = (size_t)(n_blk / 64) * (k * 64) + (size_t)k_start * 32 * 64;
                                        const int8_t *b_ptr = pw.packed_data.data() + weights_offset;

                                        // Fix: Offset compensation and scales by k_start * N_padded
                                        const int32_t *comp_ptr = pw.compensation.data() + (size_t)k_start * N_padded + n_blk;
                                        const float *scales_ptr = pw.scales.data() + (size_t)k_start * N_padded + n_blk;
                                        const float *mins_ptr = pw.mins.data() + (size_t)k_start * N_padded + n_blk;

                                        QuantisedGemmParams params;
                                        params.A = blocks;
                                        params.B_packed = b_ptr;
                                        params.comp = comp_ptr;
                                        params.scales = scales_ptr;
                                        params.mins = pw.has_mins ? mins_ptr : nullptr;
                                        params.K_blocks = k_count;
                                        params.N = 64;
                                        params.ldc = N_padded;
                                        params.bias = bias ? bias + n_blk : nullptr;
                                        params.mask = mask ? mask + i * n + n_blk : nullptr;
                                        params.A_stride = k_blocks * sizeof(Q8_1Block);

                                        bool is_tail = (n_blk + 64 > n);
                                        float C_temp[64];
                                        if (is_tail)
                                        {
                                            int valid_n = n - n_blk;
                                            std::memcpy(C_temp, C + i * n + n_blk, valid_n * sizeof(float));
                                            std::memset(C_temp + valid_n, 0, (64 - valid_n) * sizeof(float));
                                            params.C = C_temp;
                                        }
                                        else
                                        {
                                            params.C = C + i * n + n_blk;
                                        }

                                        bool is_last_k_tile = (k_start + k_count == k_blocks);
                                        bool current_do_softmax = do_softmax && is_last_k_tile;
                                        bool current_do_swiglu = do_swiglu && is_last_k_tile;

                                        float tmp_max[2], tmp_sum[2];
                                        if (current_do_softmax)
                                        {
                                            if (rows_to_process == 2)
                                            {
                                                params.local_max = tmp_max;
                                                params.local_sum = tmp_sum;
                                            }
                                            else
                                            {
                                                int block_idx = n_blk / 64;
                                                params.local_max = local_max + i * blocks_per_row + block_idx;
                                                params.local_sum = local_sum + i * blocks_per_row + block_idx;
                                            }
                                        }
                                        else
                                        {
                                            params.local_max = nullptr;
                                            params.local_sum = nullptr;
                                        }
                                        params.do_softmax = current_do_softmax;

                                        if (current_do_swiglu)
                                        {
                                            params.gate_input = gate_input + i * n + n_blk;
                                            params.do_swiglu = true;
                                        }
                                        else
                                        {
                                            params.gate_input = nullptr;
                                            params.do_swiglu = false;
                                        }

                                        if (rows_to_process == 2)
                                        {
                                            kernel_m2(&params);
                                            if (current_do_softmax)
                                            {
                                                int block_idx = n_blk / 64;
                                                local_max[i * blocks_per_row + block_idx] = tmp_max[0];
                                                local_sum[i * blocks_per_row + block_idx] = tmp_sum[0];
                                                local_max[(i + 1) * blocks_per_row + block_idx] = tmp_max[1];
                                                local_sum[(i + 1) * blocks_per_row + block_idx] = tmp_sum[1];
                                            }
                                        }
                                        else
                                            kernel(&params);

                                        if (is_tail)
                                        {
                                            int valid_n = n - n_blk;
                                            std::memcpy(C + i * n + n_blk, C_temp, valid_n * sizeof(float));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        int unroll = debugEnv().gemm.gemm_m_unroll_factor;
                        if (unroll != 1 && unroll != 2)
                            unroll = 2;

                        // Standard path: Collapse M and N for maximum parallelism
#pragma omp for collapse(2) schedule(static)
                        for (int n_task = 0; n_task < n; n_task += n_task_block)
                        {
                            for (int i = 0; i < m; i += unroll)
                            {
                                int rows_left = m - i;
                                int rows_to_process = (rows_left >= unroll) ? unroll : rows_left;
                                if (is_padded && rows_to_process > 1)
                                    rows_to_process = 1;

                                Q8_1Block *blocks = all_blocks + i * k_blocks;

                                int n_end = std::min(n, n_task + n_task_block);
                                for (int n_blk = n_task; n_blk < n_end; n_blk += 64)
                                {
                                    // Calculate packed weights offset
                                    size_t weights_offset = (size_t)(n_blk / 64) * (k * 64);
                                    const int8_t *b_ptr = pw.packed_data.data() + weights_offset;

                                    const int32_t *comp_ptr = pw.compensation.data() + n_blk;
                                    const float *scales_ptr = pw.scales.data() + n_blk;
                                    const float *mins_ptr = pw.mins.data() + n_blk;

                                    QuantisedGemmParams params;
                                    params.A = blocks;
                                    params.B_packed = b_ptr;
                                    params.comp = comp_ptr;
                                    params.scales = scales_ptr;
                                    params.mins = pw.has_mins ? mins_ptr : nullptr;
                                    params.K_blocks = k_blocks;
                                    params.N = 64;
                                    params.ldc = N_padded;
                                    params.bias = bias ? bias + n_blk : nullptr;
                                    params.mask = mask ? mask + i * n + n_blk : nullptr;
                                    params.A_stride = k_blocks * sizeof(Q8_1Block);

                                    bool is_tail = (n_blk + 64 > n);
                                    float C_temp[64];
                                    if (is_tail)
                                    {
                                        int valid_n = n - n_blk;
                                        std::memcpy(C_temp, C + i * n + n_blk, valid_n * sizeof(float));
                                        std::memset(C_temp + valid_n, 0, (64 - valid_n) * sizeof(float));
                                        params.C = C_temp;
                                    }
                                    else
                                    {
                                        params.C = C + i * n + n_blk;
                                    }

                                    float tmp_max[2], tmp_sum[2];
                                    if (do_softmax)
                                    {
                                        if (rows_to_process == 2)
                                        {
                                            params.local_max = tmp_max;
                                            params.local_sum = tmp_sum;
                                        }
                                        else
                                        {
                                            int block_idx = n_blk / 64;
                                            params.local_max = local_max + i * blocks_per_row + block_idx;
                                            params.local_sum = local_sum + i * blocks_per_row + block_idx;
                                        }
                                    }
                                    else
                                    {
                                        params.local_max = nullptr;
                                        params.local_sum = nullptr;
                                    }
                                    params.do_softmax = do_softmax;

                                    if (do_swiglu)
                                    {
                                        params.gate_input = gate_input + i * n + n_blk;
                                        params.do_swiglu = true;
                                    }
                                    else
                                    {
                                        params.gate_input = nullptr;
                                        params.do_swiglu = false;
                                    }

                                    if (rows_to_process == 2)
                                    {
                                        kernel_m2(&params);
                                        if (do_softmax)
                                        {
                                            int block_idx = n_blk / 64;
                                            local_max[i * blocks_per_row + block_idx] = tmp_max[0];
                                            local_sum[i * blocks_per_row + block_idx] = tmp_sum[0];
                                            local_max[(i + 1) * blocks_per_row + block_idx] = tmp_max[1];
                                            local_sum[(i + 1) * blocks_per_row + block_idx] = tmp_sum[1];
                                        }
                                    }
                                    else
                                        kernel(&params);

                                    if (is_tail)
                                    {
                                        int valid_n = n - n_blk;
                                        std::memcpy(C + i * n + n_blk, C_temp, valid_n * sizeof(float));
                                    }
                                }
                            }
                        }
                    }
                }; // end do_multiply_work lambda

                // Execute using OMP_WORKSHARE_REGION for nested-safe parallelism
                OMP_WORKSHARE_REGION(do_multiply_work);

                return true;
            }

            /**
             * @brief Activation-activation multiply with packed weight fallback
             *
             * When B is nullptr, falls back to multiply() which uses packed weights.
             * This allows the pipeline to use multiply_activations uniformly for both
             * weight projections (Q/K/V) and attention (Q@K^T, scores@V).
             *
             * @note The `transpose_B` parameter is ignored for weight projections
             *       (B=nullptr) since weights are always pre-transposed during packing.
             */
            bool multiply_activations(const float *A, const float *B, float *C, int m, int n, int k, bool transpose_B, float alpha, float beta, const MPIContext *ctx, int device_idx) override
            {
                // When B is nullptr, use packed weights via multiply()
                if (!B)
                {
                    // Note: transpose_B is ignored - packed weights are always in the correct layout
                    // beta > 0 means accumulate into existing C
                    bool accumulate = (beta > 0.0f);
                    return multiply(A, C, m, n, k, accumulate, alpha, beta, ctx, device_idx);
                }

                // Activation-activation GEMM not supported - use FloatingPointGemmKernel
                LOG_ERROR("[QuantisedGemmKernel] multiply_activations with two activation matrices not supported");
                return false;
            }

            /**
             * @brief Strided activation-activation multiply (not supported for quantized kernel)
             *
             * QuantisedGemmKernel is designed for weight projections (packed INT8 weights).
             * For activation-activation GEMM (attention Q@K^T, scores@V), use FloatingPointGemmKernel.
             */
            bool multiply_activations_strided(const float *A, const float *B, float *C, int m, int n, int k, int stride_a, int stride_b, int stride_c, bool transpose_B, float alpha, float beta, const MPIContext *ctx, int device_idx) override
            {
                (void)A;
                (void)B;
                (void)C;
                (void)m;
                (void)n;
                (void)k;
                (void)stride_a;
                (void)stride_b;
                (void)stride_c;
                (void)transpose_B;
                (void)alpha;
                (void)beta;
                (void)ctx;
                (void)device_idx;
                LOG_ERROR("[QuantisedGemmKernel] multiply_activations_strided not supported - use FloatingPointGemmKernel");
                return false;
            }

            /**
             * @brief Strided GEMM + softmax for Q8_1 x Q8_1 attention (Q @ K^T)
             *
             * Optimized for attention score computation where Q and K are Q8_1 tensors
             * with per-head striding. Computes:
             *   C = softmax(scale * Q @ K^T + mask)
             *
             * @param A Q tensor head pointer (Q8_1 blocks)
             * @param B K tensor head pointer (Q8_1 blocks)
             * @param C Output attention scores (FP32)
             * @param m seq_len (number of query positions)
             * @param n kv_len (number of key positions)
             * @param k head_dim (must be multiple of 32)
             * @param lda Stride between Q rows in elements (n_heads * head_dim)
             * @param ldb Stride between K rows in elements (n_kv_heads * head_dim)
             * @param ldc Stride of output (typically kv_len)
             * @param scale Attention scale (1/sqrt(d_k))
             * @param transpose_B Must be true (K^T)
             * @param softmax_axis 1 (row-wise)
             * @param mask Optional additive mask
             * @param is_causal Apply causal masking
             * @param mpi_ctx MPI context (unused)
             * @param device_idx Device index (-1 for CPU)
             * @param format_A Activation format (must be Q8_1)
             * @param format_B Activation format (must be Q8_1)
             */
            bool multiply_with_softmax_strided_typed_impl(
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
                ActivationFormat format_B) override
            {
                (void)mpi_ctx;
                (void)device_idx;
                (void)softmax_axis; // Always row-wise

                // Validate inputs
                if (!transpose_B)
                {
                    LOG_ERROR("[QuantisedGemmKernel] multiply_with_softmax_strided_typed_impl: transpose_B must be true for attention");
                    return false;
                }

                // Check for Q8_1 format
                if (format_A != ActivationFormat::Q8_1 || format_B != ActivationFormat::Q8_1)
                {
                    LOG_ERROR("[QuantisedGemmKernel] multiply_with_softmax_strided_typed_impl: requires Q8_1 format, got A="
                              << static_cast<int>(format_A) << " B=" << static_cast<int>(format_B));
                    return false;
                }

                // Convert element strides to byte strides for Q8_1
                // lda/ldb are in elements (floats), but Q8_1 packs 32 elements per block
                // So stride in blocks = stride_elements / 32, stride in bytes = blocks * sizeof(Q8_1Block)
                int k_blocks = (k + 31) / 32;

                // For attention: Q is [seq_len, n_heads, head_dim], K is [kv_len, n_kv_heads, head_dim]
                // lda = n_heads * head_dim (elements), ldb = n_kv_heads * head_dim (elements)
                // In Q8_1 representation: stride = (n_heads * head_dim) / 32 * sizeof(Q8_1Block)
                // But head_dim must be multiple of 32, so: stride = n_heads * k_blocks * sizeof(Q8_1Block)

                int stride_q_bytes = (lda / k) * k_blocks * static_cast<int>(sizeof(Q8_1Block));
                int stride_k_bytes = (ldb / k) * k_blocks * static_cast<int>(sizeof(Q8_1Block));

                return compute_q8_1_strided_gemm_softmax(
                    A, B, C,
                    m, n, k,
                    stride_q_bytes, stride_k_bytes,
                    scale, mask, is_causal, 0); // causal_offset=0 for prefill
            }

            // =============================================================================
            // Fused Multi-GEMM Interface Implementation (Activation Sharing)
            // =============================================================================

            /**
             * @brief Check if this kernel supports activation sharing
             */
            bool supports_activation_sharing() const override
            {
                return true; // QuantisedGemmKernel supports Q8_1 activation sharing
            }

            /**
             * @brief Get buffer size needed for quantized activations
             */
            size_t get_quantized_activation_buffer_size(int m, int k) const override
            {
                int k_blocks = (k + 31) / 32;
                return static_cast<size_t>(m) * k_blocks * sizeof(Q8_1Block);
            }

            /**
             * @brief Quantize FP32 activations to Q8_1 for reuse across multiple GEMMs
             *
             * This is the first step in the fused multi-GEMM workflow.
             * The resulting Q8_1 blocks can be passed to multiply_with_precomputed_q8_1()
             * multiple times without redundant quantization.
             *
             * Uses SIMD-optimized quantization (AVX-512/AVX2) via shared quantization utility.
             */
            bool quantize_activations(
                const float *A,
                void *q8_1_buffer,
                int m, int k) override
            {
                // Delegate to shared quantization utility
                return quantization::quantize_fp32_to_q8_1_buffer(A, q8_1_buffer, m, k);
            }

            /**
             * @brief GEMM with pre-quantized Q8_1 activations and fused post-ops
             *
             * This is the second step in the fused multi-GEMM workflow.
             * Uses pre-quantized activations from quantize_activations(),
             * eliminating redundant FP32→Q8_1 conversion.
             *
             * Supports fused operations via GemmFusedOps:
             * - SwiGLU: output *= swish(gate)
             * - Softmax: output = softmax(output * scale + mask)
             *
             * MPI Tensor Parallelism:
             * When mpi_ctx is provided with world_size > 1, this function uses
             * column-parallel execution: each rank computes only its slice of
             * output columns, reducing memory bandwidth (but not compute, since
             * weights are replicated). For true compute savings, use weight sharding.
             */
            bool multiply_with_precomputed_q8_1(
                const void *q8_1_activations,
                float *C,
                int m, int n, int k,
                const TensorBase *bias,
                bool accumulate,
                float alpha, float beta,
                const MPIContext *mpi_ctx,
                int device_idx,
                const GemmFusedOps &fused_ops,
                DeviceWorkspaceManager *workspace = nullptr) override
            {
                (void)device_idx;
                (void)alpha;     // Currently assumed 1.0
                (void)workspace; // CPU kernel doesn't need external workspace

                if (!q8_1_activations || !C)
                    return false;

                // Extract raw bias pointer from TensorBase
                const float *bias_ptr = bias ? bias->fp32_data() : nullptr;

                // DEBUG: Log thread state at entry point
                if (n == 896 && k == 896)
                {
                    LOG_TRACE("[ENTRY] tid=" << omp_get_thread_num()
                                             << " num_threads=" << omp_get_num_threads()
                                             << " in_parallel=" << omp_in_parallel()
                                             << " C=" << static_cast<const void *>(C));
                }

                // Get reference to packed weights (either external or owned)
                const auto &pw = packed_weights();

                // Debug: log packed weights pointer for comparison
                if (n == 896 && k == 896)
                {
                    // Log q8_1_activations pointer and first block for comparison
                    const Q8_1Block *blocks = static_cast<const Q8_1Block *>(q8_1_activations);

                    // Compute a checksum of all Q8_1 blocks to detect any differences
                    int blocks_per_row = k / 32;
                    int num_blocks = m * blocks_per_row;
                    float d_sum = 0.0f;
                    int32_t sum_qs_sum = 0;
                    uint64_t qs_checksum = 0;
                    for (int i = 0; i < num_blocks; ++i)
                    {
                        d_sum += blocks[i].d;
                        sum_qs_sum += blocks[i].sum_qs;
                        for (int j = 0; j < 32; ++j)
                        {
                            qs_checksum += static_cast<uint64_t>(static_cast<uint8_t>(blocks[i].qs[j])) * (i * 32 + j + 1);
                        }
                    }

                    // Compute a FULL checksum of the packed weights data
                    uint64_t pw_checksum = 0;
                    size_t pw_size = pw.packed_data.size();
                    const uint8_t *pw_data = reinterpret_cast<const uint8_t *>(pw.packed_data.data());
                    for (size_t i = 0; i < pw_size; i += 997)
                    { // Sample every 997th byte for speed
                        pw_checksum += static_cast<uint64_t>(pw_data[i]) * (i + 1);
                    }

                    LOG_TRACE("[multiply_with_precomputed_q8_1] m=" << m << " n=" << n << " k=" << k
                                                                    << " this=" << static_cast<const void *>(this)
                                                                    << " C=" << static_cast<const void *>(C)
                                                                    << " q8_1_act=" << static_cast<const void *>(q8_1_activations)
                                                                    << " num_blocks=" << num_blocks
                                                                    << " d_sum=" << d_sum
                                                                    << " sum_qs_sum=" << sum_qs_sum
                                                                    << " qs_checksum=" << qs_checksum
                                                                    << " block0.d=" << blocks[0].d
                                                                    << " block0.sum_qs=" << blocks[0].sum_qs
                                                                    << " block0.qs[0:4]=" << (int)blocks[0].qs[0] << "," << (int)blocks[0].qs[1] << "," << (int)blocks[0].qs[2] << "," << (int)blocks[0].qs[3]
                                                                    << " block1.d=" << (num_blocks > 1 ? blocks[1].d : 0)
                                                                    << " block" << (blocks_per_row - 1) << ".d=" << blocks[blocks_per_row - 1].d     // Last block of row 0
                                                                    << " block" << blocks_per_row << ".d=" << (m > 1 ? blocks[blocks_per_row].d : 0) // First block of row 1 (if exists)
                                                                    << " pw.packed_data.data()=" << static_cast<const void *>(pw.packed_data.data())
                                                                    << " pw_size=" << pw_size
                                                                    << " pw_FULL_checksum=" << pw_checksum
                                                                    << " pw.N=" << pw.N << " pw.K=" << pw.K
                                                                    << " fused_swiglu=" << fused_ops.is_swiglu()
                                                                    << " fused_softmax=" << fused_ops.is_softmax()
                                                                    << " accumulate=" << accumulate
                                                                    << " beta=" << beta);
                }

                // Check dimensions against packed weights
                if (n != pw.N || k != pw.K)
                {
                    LOG_ERROR("Dimension mismatch in multiply_with_precomputed_q8_1: "
                              << "expected n=" << pw.N << ", k=" << pw.K
                              << " got n=" << n << ", k=" << k);
                    return false;
                }

                // Extract fused op parameters
                const float *gate_input = fused_ops.is_swiglu() ? fused_ops.gate_input : nullptr;
                bool do_swiglu = fused_ops.is_swiglu();
                bool do_softmax = fused_ops.is_softmax();
                const float *softmax_mask = fused_ops.softmax_mask;
                float *local_max = fused_ops.online_max;
                float *local_sum = fused_ops.online_sum;

                // Get JIT kernels
                static QuantisedGemmJit_M1 jit;
                static QuantisedGemmJit_M2 jit_m2;
                auto kernel = jit.get_kernel();
                auto kernel_m2 = jit_m2.get_kernel();

                // Determine beta handling mode
                bool needs_zero = (!accumulate || beta == 0.0f);
                bool needs_scale = (!needs_zero && beta != 1.0f);

                int k_blocks = k / 32;
                int N_padded = ((n + 63) / 64) * 64; // Stride for compensation/scales arrays
                const Q8_1Block *all_blocks = static_cast<const Q8_1Block *>(q8_1_activations);

                // OMP_WORKSHARE_REGION handles nested parallelism automatically
                auto do_gemm_work = [&]()
                {
                    // FUSED: Zero/scale C inside the same parallel region as GEMM
                    if (needs_zero)
                    {
#pragma omp for schedule(static) nowait
                        for (int i = 0; i < m; ++i)
                        {
                            std::memset(C + i * n, 0, n * sizeof(float));
                        }
                    }
                    else if (needs_scale)
                    {
#pragma omp for schedule(static) nowait
                        for (int i = 0; i < m; ++i)
                        {
                            float *row = C + i * n;
                            for (int j = 0; j < n; ++j)
                                row[j] *= beta;
                        }
                    }

                    // Barrier to ensure zeroing complete before GEMM writes
#pragma omp barrier

                    // DEBUG: Check C[0:4] after zeroing (only thread 0 to avoid race)
                    if (n == 896 && k == 896 && omp_get_thread_num() == 0)
                    {
                        LOG_TRACE("[AFTER_ZERO] C[0:4]=" << C[0] << "," << C[1] << "," << C[2] << "," << C[3]
                                                         << " needs_zero=" << needs_zero
                                                         << " omp_in_parallel=" << omp_in_parallel()
                                                         << " num_threads=" << omp_get_num_threads());
                    }

                    // Cache-aware blocking (same logic as other paths)
                    int num_threads = omp_get_num_threads();
                    long long l2_size = cpu_l2_cache_size();
                    if (l2_size == 0)
                        l2_size = 1024 * 1024;

                    long long l3_size = cpu_l3_cache_size();
                    if (l3_size == 0)
                        l3_size = l2_size * num_threads;

                    long long l2_limit = (long long)(l2_size * 0.9);
                    long long l3_limit_per_thread = (long long)(l3_size * 0.9 / num_threads);
                    long long block_size_limit = std::min(l2_limit, l3_limit_per_thread);
                    if (block_size_limit < 65536)
                        block_size_limit = 65536;

                    int max_n_block = block_size_limit / k;
                    max_n_block = (max_n_block / 64) * 64;
                    if (max_n_block < 64)
                        max_n_block = 64;

                    int target_tasks = num_threads * 4;
                    int m_tasks = (m + 1) / 2;
                    if (m_tasks < 1)
                        m_tasks = 1;

                    int needed_n_tasks = (target_tasks + m_tasks - 1) / m_tasks;
                    if (needed_n_tasks < 1)
                        needed_n_tasks = 1;

                    int calc_block = (n + needed_n_tasks - 1) / needed_n_tasks;
                    calc_block = (calc_block + 63) / 64 * 64;
                    if (calc_block < 64)
                        calc_block = 64;

                    int n_task_block;
                    if (calc_block > max_n_block)
                    {
                        int num_splits = (n + max_n_block - 1) / max_n_block;
                        int even_block = (n + num_splits - 1) / num_splits;
                        even_block = (even_block + 63) / 64 * 64;
                        n_task_block = even_block;
                    }
                    else
                    {
                        n_task_block = calc_block;
                    }

                    int unroll = debugEnv().gemm.gemm_m_unroll_factor;
                    if (unroll != 1 && unroll != 2)
                        unroll = 2;

#pragma omp for collapse(2) schedule(static)
                    for (int n_task = 0; n_task < n; n_task += n_task_block)
                    {
                        for (int i = 0; i < m; i += unroll)
                        {
                            int rows_left = m - i;
                            int rows_to_process = (rows_left >= unroll) ? unroll : rows_left;

                            // Pointer to Q8_1 blocks for this row
                            const Q8_1Block *blocks = all_blocks + i * k_blocks;

                            int n_end = std::min(n, n_task + n_task_block);
                            for (int n_blk = n_task; n_blk < n_end; n_blk += 64)
                            {
                                size_t weights_offset = (size_t)(n_blk / 64) * (k * 64);
                                const int8_t *b_ptr = pw.packed_data.data() + weights_offset;

                                const int32_t *comp_ptr = pw.compensation.data() + n_blk;
                                const float *scales_ptr = pw.scales.data() + n_blk;
                                const float *mins_ptr = pw.mins.data() + n_blk;

                                volatile QuantisedGemmParams params_v;
                                params_v.A = blocks;
                                params_v.B_packed = b_ptr;
                                params_v.comp = comp_ptr;
                                params_v.scales = scales_ptr;
                                params_v.mins = pw.has_mins ? mins_ptr : nullptr;
                                params_v.C = C + i * n + n_blk;
                                params_v.K_blocks = k_blocks;
                                params_v.N = 64;
                                params_v.ldc = N_padded; // Stride for comp/scales arrays, NOT C output
                                params_v.bias = bias_ptr ? bias_ptr + n_blk : nullptr;
                                params_v.mask = softmax_mask ? (softmax_mask + i * n + n_blk) : nullptr;
                                params_v.A_stride = k_blocks * sizeof(Q8_1Block);
                                params_v.local_max = local_max;
                                params_v.local_sum = local_sum;
                                params_v.do_softmax = do_softmax;
                                params_v.gate_input = gate_input ? (gate_input + i * n + n_blk) : nullptr;
                                params_v.do_swiglu = do_swiglu;

                                // DEBUG: Log first JIT call params for n_blk=0, i=0
                                if (n == 896 && k == 896 && n_blk == 0 && i == 0)
                                {
                                    uint64_t a_cksum = 0, b_cksum = 0;
                                    const uint8_t *a_bytes = static_cast<const uint8_t *>(params_v.A);
                                    const uint8_t *b_bytes = static_cast<const uint8_t *>(params_v.B_packed);
                                    for (int x = 0; x < 64; ++x)
                                    {
                                        a_cksum += static_cast<uint64_t>(a_bytes[x]) * (x + 1);
                                        b_cksum += static_cast<uint64_t>(b_bytes[x]) * (x + 1);
                                    }

                                    // Checksum comp array data (K_blocks floats)
                                    uint64_t comp_cksum = 0;
                                    if (params_v.comp)
                                    {
                                        const uint8_t *comp_bytes = reinterpret_cast<const uint8_t *>(params_v.comp);
                                        for (int x = 0; x < static_cast<int>(params_v.K_blocks) * sizeof(float); ++x)
                                        {
                                            comp_cksum += static_cast<uint64_t>(comp_bytes[x]) * (x + 1);
                                        }
                                    }

                                    // Checksum scales array data (K_blocks floats)
                                    uint64_t scales_cksum = 0;
                                    if (params_v.scales)
                                    {
                                        const uint8_t *scales_bytes = reinterpret_cast<const uint8_t *>(params_v.scales);
                                        for (int x = 0; x < static_cast<int>(params_v.K_blocks) * sizeof(float); ++x)
                                        {
                                            scales_cksum += static_cast<uint64_t>(scales_bytes[x]) * (x + 1);
                                        }
                                    }

                                    // Also dump first 36 bytes of A in hex (first Q8_1 block = 36 bytes)
                                    std::ostringstream hex_dump;
                                    for (int x = 0; x < 36; ++x)
                                    {
                                        hex_dump << std::hex << std::setfill('0') << std::setw(2) << (int)a_bytes[x];
                                        if (x % 4 == 3)
                                            hex_dump << " ";
                                    }

                                    LOG_TRACE("[JIT_PARAMS] n_blk=" << n_blk << " i=" << i
                                                                    << " A_ptr=" << static_cast<const void *>(params_v.A)
                                                                    << " B_ptr=" << static_cast<const void *>(params_v.B_packed)
                                                                    << " C_ptr=" << static_cast<const void *>(params_v.C)
                                                                    << " A_64byte_cksum=" << a_cksum
                                                                    << " B_64byte_cksum=" << b_cksum
                                                                    << " C_before=" << params_v.C[0] << "," << params_v.C[1]
                                                                    << " A_hex=" << hex_dump.str()
                                                                    << " output_format=" << static_cast<int>(params_v.output_format)
                                                                    << " do_softmax=" << params_v.do_softmax
                                                                    << " do_swiglu=" << params_v.do_swiglu
                                                                    << " K_blocks=" << params_v.K_blocks
                                                                    << " N=" << params_v.N
                                                                    << " ldc=" << params_v.ldc
                                                                    << " A_stride=" << params_v.A_stride
                                                                    << " comp=" << static_cast<const void *>(params_v.comp)
                                                                    << " comp_data_cksum=" << comp_cksum
                                                                    << " scales=" << static_cast<const void *>(params_v.scales)
                                                                    << " scales_data_cksum=" << scales_cksum
                                                                    << " mins=" << static_cast<const void *>(params_v.mins)
                                                                    << " bias=" << static_cast<const void *>(params_v.bias)
                                                                    << " mask=" << static_cast<const void *>(params_v.mask)
                                                                    << " local_max=" << static_cast<const void *>(params_v.local_max)
                                                                    << " local_sum=" << static_cast<const void *>(params_v.local_sum)
                                                                    << " gate_input=" << static_cast<const void *>(params_v.gate_input)
                                                                    << " C_q8_1=" << static_cast<const void *>(params_v.C_q8_1)
                                                                    << " C_q8_1_stride=" << params_v.C_q8_1_stride);
                                }

                                // Copy to non-volatile for kernel call
                                QuantisedGemmParams params;
                                std::memcpy(&params, const_cast<QuantisedGemmParams *>(&params_v), sizeof(params));

                                // Memory barrier to prevent reordering
                                asm volatile("" ::: "memory");

                                // DEBUG: Log params struct address for n_blk=0 i=0
                                if (n == 896 && k == 896 && n_blk == 0 && i == 0)
                                {
                                    LOG_TRACE("[PARAMS_ADDR] &params=" << static_cast<const void *>(&params)
                                                                       << " alignment=" << (reinterpret_cast<uintptr_t>(&params) % 64)
                                                                       << " sizeof(params)=" << sizeof(params)
                                                                       << " kernel_ptr=" << reinterpret_cast<const void *>(kernel));
                                }

                                if (rows_to_process == 2)
                                    kernel_m2(&params);
                                else
                                    kernel(&params);

                                // DEBUG: Immediately log output after kernel returns for n_blk=0 i=0
                                if (n == 896 && k == 896 && n_blk == 0 && i == 0)
                                {
                                    LOG_TRACE("[KERNEL_DONE_IMMEDIATE] C[0:4]="
                                              << std::setprecision(10) << params.C[0] << "," << params.C[1] << "," << params.C[2] << "," << params.C[3]
                                              << " tid=" << omp_get_thread_num());
                                }
                            }
                        }
                    }
                }; // end do_gemm_work lambda

                OMP_WORKSHARE_REGION(do_gemm_work);

                // CRITICAL: Ensure all memory writes are visible before logging/returning
                // The omp for inside do_gemm_work has implicit barrier, but if we're inside
                // an existing parallel region (omp_in_parallel()==true), there's no barrier
                // after work_fn() returns from OMP_WORKSHARE_REGION.
                std::atomic_thread_fence(std::memory_order_seq_cst);

                // Debug: log output after GEMM completes
                if (n == 896 && k == 896)
                {
                    LOG_TRACE("[multiply_with_precomputed_q8_1 DONE] C[0:4]="
                              << std::setprecision(10) << C[0] << "," << C[1] << "," << C[2] << "," << C[3]);
                }

                return true;
            }

            /**
             * @brief GEMM with pre-computed Q8_1 activations, outputting to Q8_1 format
             *
             * This method combines quantized GEMM with fused Q8_1 requantization of the output.
             * Instead of storing FP32 results, it directly outputs Q8_1 blocks.
             *
             * Use case: When multiple GEMMs share the same input (e.g., Q/K/V projections),
             * and the output will be used as input to another quantized kernel:
             *   1. Call quantize_activations() once on FP32 input
             *   2. Call multiply_with_precomputed_q8_1_to_q8_1() for each projection
             *   3. Use Q8_1 outputs directly in subsequent kernels
             *
             * @param q8_1_activations Pre-quantized Q8_1 activation blocks from quantize_activations()
             * @param C_q8_1 Output Q8_1 blocks [m, (n+31)/32] - must be pre-allocated
             * @param m Number of rows (sequence length)
             * @param n Output features (columns, will be rounded up to 32)
             * @param k Input features (rows in weight matrix)
             * @param bias Optional bias vector [n] to add before requantization (nullptr if none)
             * @param accumulate If true, quantizes FP32 result first then adds (not recommended)
             * @param mpi_ctx MPI context
             * @param device_idx Device index
             *
             * @return true on success
             *
             * @note Output n is rounded up to nearest multiple of 32 for Q8_1 alignment.
             *       Each output row produces ceil(n/32) Q8_1 blocks.
             * @note Bias is added to the FP32 GEMM result before Q8_1 requantization.
             */
            bool multiply_with_precomputed_q8_1_to_q8_1(
                const void *q8_1_activations,
                void *C_q8_1,
                int m, int n, int k,
                const TensorBase *bias = nullptr,
                bool accumulate = false,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                DeviceWorkspaceManager *workspace = nullptr) override
            {
                (void)device_idx;
                (void)accumulate; // Q8_1 output doesn't support accumulate (would need dequant+add+requant)
                (void)mpi_ctx;
                (void)workspace; // CPU kernel doesn't need external workspace

                if (!q8_1_activations || !C_q8_1)
                    return false;

                // Extract raw bias pointer from TensorBase
                const float *bias_ptr = bias ? bias->fp32_data() : nullptr;

                // Get reference to packed weights
                const auto &pw = packed_weights();

                // Check dimensions
                if (n != pw.N || k != pw.K)
                {
                    LOG_ERROR("Dimension mismatch in multiply_with_precomputed_q8_1_to_q8_1: "
                              << "expected n=" << pw.N << ", k=" << pw.K
                              << " got n=" << n << ", k=" << k);
                    return false;
                }

                // Get JIT kernels (with Q8_1 output support)
                static QuantisedGemmJit_M1 jit;
                static QuantisedGemmJit_M2 jit_m2;
                auto kernel = jit.get_kernel();
                auto kernel_m2 = jit_m2.get_kernel();

                int k_blocks = k / 32;
                int N_padded = ((n + 63) / 64) * 64; // Stride for compensation/scales arrays
                const Q8_1Block *all_blocks = static_cast<const Q8_1Block *>(q8_1_activations);
                Q8_1Block *output_blocks = static_cast<Q8_1Block *>(C_q8_1);

                // Calculate output stride: blocks per row = ceil(n/32)
                const int output_blocks_per_row = (n + 31) / 32;
                const size_t output_stride = output_blocks_per_row * sizeof(Q8_1Block);

                // OMP_WORKSHARE_REGION handles nested parallelism automatically
                auto do_gemm_q8_1_work = [&]()
                {
                    // Cache-aware blocking
                    int num_threads = omp_get_num_threads();
                    long long l2_size = cpu_l2_cache_size();
                    if (l2_size == 0)
                        l2_size = 1024 * 1024;

                    long long l3_size = cpu_l3_cache_size();
                    if (l3_size == 0)
                        l3_size = l2_size * num_threads;

                    long long l2_limit = (long long)(l2_size * 0.9);
                    long long l3_limit_per_thread = (long long)(l3_size * 0.9 / num_threads);
                    long long block_size_limit = std::min(l2_limit, l3_limit_per_thread);
                    if (block_size_limit < 65536)
                        block_size_limit = 65536;

                    int max_n_block = block_size_limit / k;
                    max_n_block = (max_n_block / 64) * 64;
                    if (max_n_block < 64)
                        max_n_block = 64;

                    int target_tasks = num_threads * 4;
                    int m_tasks = (m + 1) / 2;
                    if (m_tasks < 1)
                        m_tasks = 1;

                    int needed_n_tasks = (target_tasks + m_tasks - 1) / m_tasks;
                    if (needed_n_tasks < 1)
                        needed_n_tasks = 1;

                    int calc_block = (n + needed_n_tasks - 1) / needed_n_tasks;
                    calc_block = (calc_block + 63) / 64 * 64;
                    if (calc_block < 64)
                        calc_block = 64;

                    int n_task_block;
                    if (calc_block > max_n_block)
                    {
                        int num_splits = (n + max_n_block - 1) / max_n_block;
                        int even_block = (n + num_splits - 1) / num_splits;
                        even_block = (even_block + 63) / 64 * 64;
                        n_task_block = even_block;
                    }
                    else
                    {
                        n_task_block = calc_block;
                    }

                    int unroll = debugEnv().gemm.gemm_m_unroll_factor;
                    if (unroll != 1 && unroll != 2)
                        unroll = 2;

#pragma omp for collapse(2) schedule(static)
                    for (int n_task = 0; n_task < n; n_task += n_task_block)
                    {
                        for (int i = 0; i < m; i += unroll)
                        {
                            int rows_left = m - i;
                            int rows_to_process = (rows_left >= unroll) ? unroll : rows_left;

                            // Pointer to Q8_1 blocks for this row
                            const Q8_1Block *blocks = all_blocks + i * k_blocks;

                            int n_end = std::min(n, n_task + n_task_block);
                            for (int n_blk = n_task; n_blk < n_end; n_blk += 64)
                            {
                                size_t weights_offset = (size_t)(n_blk / 64) * (k * 64);
                                const int8_t *b_ptr = pw.packed_data.data() + weights_offset;

                                const int32_t *comp_ptr = pw.compensation.data() + n_blk;
                                const float *scales_ptr = pw.scales.data() + n_blk;
                                const float *mins_ptr = pw.mins.data() + n_blk;

                                volatile QuantisedGemmParams params_v;
                                params_v.A = blocks;
                                params_v.B_packed = b_ptr;
                                params_v.comp = comp_ptr;
                                params_v.scales = scales_ptr;
                                params_v.mins = pw.has_mins ? mins_ptr : nullptr;
                                params_v.C = nullptr; // FP32 output not used
                                params_v.K_blocks = k_blocks;
                                params_v.N = 64;
                                params_v.ldc = N_padded; // Stride for comp/scales arrays, NOT C output
                                // Bias is added before Q8_1 requantization (if provided)
                                params_v.bias = bias_ptr ? (bias_ptr + n_blk) : nullptr;
                                params_v.mask = nullptr;
                                params_v.A_stride = k_blocks * sizeof(Q8_1Block);
                                params_v.local_max = nullptr;
                                params_v.local_sum = nullptr;
                                params_v.do_softmax = false;
                                params_v.gate_input = nullptr;
                                params_v.do_swiglu = false;

                                // Q8_1 output parameters
                                params_v.output_format = GemmOutputFormat::Q8_1;

                                // Handle tail case: N < 64 or remaining columns < 64
                                // JIT kernel always writes 64 columns (2 Q8_1 blocks per row)
                                // Use temp buffer if we don't have space for full 64 columns
                                bool is_tail = (n_blk + 64 > n);
                                Q8_1Block temp_blocks[4]; // 2 blocks per row * 2 rows max

                                if (is_tail)
                                {
                                    // Write to temporary buffer, then copy valid blocks
                                    params_v.C_q8_1 = reinterpret_cast<uint8_t *>(temp_blocks);
                                    params_v.C_q8_1_stride = 2 * sizeof(Q8_1Block); // Always 2 blocks per row in temp
                                }
                                else
                                {
                                    // Direct write: output block pointer at row i, starting at n_blk/32 blocks
                                    params_v.C_q8_1 = reinterpret_cast<uint8_t *>(
                                        output_blocks + i * output_blocks_per_row + (n_blk / 32));
                                    params_v.C_q8_1_stride = output_stride;
                                }

                                // Copy to non-volatile for kernel call
                                QuantisedGemmParams params;
                                std::memcpy(&params, const_cast<QuantisedGemmParams *>(&params_v), sizeof(params));

                                // Memory barrier to prevent reordering
                                asm volatile("" ::: "memory");

                                if (rows_to_process == 2)
                                    kernel_m2(&params);
                                else
                                    kernel(&params);

                                // Copy valid blocks from temp buffer for tail case
                                if (is_tail)
                                {
                                    int valid_cols = n - n_blk;
                                    int valid_blocks = (valid_cols + 31) / 32; // 1 or 2 blocks
                                    int start_block = n_blk / 32;

                                    for (int r = 0; r < rows_to_process; ++r)
                                    {
                                        for (int b = 0; b < valid_blocks; ++b)
                                        {
                                            if (start_block + b < output_blocks_per_row)
                                            {
                                                output_blocks[(i + r) * output_blocks_per_row + start_block + b] =
                                                    temp_blocks[r * 2 + b];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }; // end do_gemm_q8_1_work lambda

                OMP_WORKSHARE_REGION(do_gemm_q8_1_work);

                return true;
            }

            /**
             * @brief GEMM with pre-quantized Q8_1 input, outputting to Q16_1 format
             *
             * Performs high-precision quantized GEMM where output is Q16_1 blocks.
             * This provides 256× more precision than Q8_1 output, critical for
             * preserving dynamic range in K projections for HybridQ16 attention.
             *
             * Key insight: K projection often has high dynamic range (~130 max).
             * Q8_1 output loses small values due to coarse quantization step (max/127).
             * Q16_1 uses max/16383, preserving values 128× smaller.
             *
             * @param q8_1_activations Pre-quantized Q8_1 activation blocks from quantize_activations()
             * @param C_q16_1 Output Q16_1 blocks [m, n/block_size] - must be pre-allocated
             * @param m Number of rows (sequence length)
             * @param n Output features (must be divisible by q16_block_size)
             * @param k Input features
             * @param q16_block_size Q16_1 block size: 64 for head_dim=64, 128 for head_dim=128
             * @param bias Optional bias vector [n] to add before requantization
             * @param mpi_ctx MPI context
             * @param device_idx Device index
             *
             * @return true on success
             */
            bool multiply_with_precomputed_q8_1_to_q16_1(
                const void *q8_1_activations,
                void *C_q16_1,
                int m, int n, int k,
                int q16_block_size = 64,
                const TensorBase *bias = nullptr,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                DeviceWorkspaceManager *workspace = nullptr) override
            {
                (void)device_idx;
                (void)mpi_ctx;
                (void)workspace; // CPU kernel doesn't need external workspace

                if (!q8_1_activations || !C_q16_1)
                    return false;

                // Extract raw bias pointer from TensorBase
                const float *bias_ptr = bias ? bias->fp32_data() : nullptr;

                // Validate block size
                if (q16_block_size != 32 && q16_block_size != 64 && q16_block_size != 128)
                {
                    LOG_ERROR("Invalid q16_block_size " << q16_block_size << ", must be 32, 64, or 128");
                    return false;
                }

                // Validate n is divisible by block_size
                if (n % q16_block_size != 0)
                {
                    LOG_ERROR("n=" << n << " must be divisible by q16_block_size=" << q16_block_size);
                    return false;
                }

                // Get reference to packed weights
                const auto &pw = packed_weights();

                // Check dimensions
                if (n != pw.N || k != pw.K)
                {
                    LOG_ERROR("Dimension mismatch in multiply_with_precomputed_q8_1_to_q16_1: "
                              << "expected n=" << pw.N << ", k=" << pw.K
                              << " got n=" << n << ", k=" << k);
                    return false;
                }

                // Get JIT kernels (with Q16_1 output support)
                static QuantisedGemmJit_M1 jit;
                static QuantisedGemmJit_M2 jit_m2;
                auto kernel = jit.get_kernel();
                auto kernel_m2 = jit_m2.get_kernel();

                int k_blocks = k / 32;
                int N_padded = ((n + 63) / 64) * 64;
                const Q8_1Block *all_blocks = static_cast<const Q8_1Block *>(q8_1_activations);

                // Calculate output Q16_1 block parameters
                // Block sizes: 32 → 72 bytes, 64 → 136 bytes, 128 → 264 bytes
                const int q16_block_bytes = 8 + 2 * q16_block_size; // d(4) + sum_qs(4) + qs[block_size]*2
                const int output_blocks_per_row = n / q16_block_size;
                const size_t output_stride = output_blocks_per_row * q16_block_bytes;

                auto do_gemm_q16_1_work = [&]()
                {
                    int num_threads = omp_get_num_threads();
                    long long l2_size = cpu_l2_cache_size();
                    if (l2_size == 0)
                        l2_size = 1024 * 1024;

                    long long l3_size = cpu_l3_cache_size();
                    if (l3_size == 0)
                        l3_size = l2_size * num_threads;

                    long long l2_limit = (long long)(l2_size * 0.9);
                    long long l3_limit_per_thread = (long long)(l3_size * 0.9 / num_threads);
                    long long block_size_limit = std::min(l2_limit, l3_limit_per_thread);
                    if (block_size_limit < 65536)
                        block_size_limit = 65536;

                    int max_n_block = block_size_limit / k;
                    max_n_block = (max_n_block / 64) * 64;
                    if (max_n_block < 64)
                        max_n_block = 64;

                    int target_tasks = num_threads * 4;
                    int m_tasks = (m + 1) / 2;
                    if (m_tasks < 1)
                        m_tasks = 1;

                    int needed_n_tasks = (target_tasks + m_tasks - 1) / m_tasks;
                    if (needed_n_tasks < 1)
                        needed_n_tasks = 1;

                    int calc_block = (n + needed_n_tasks - 1) / needed_n_tasks;
                    calc_block = (calc_block + 63) / 64 * 64;
                    if (calc_block < 64)
                        calc_block = 64;

                    int n_task_block;
                    if (calc_block > max_n_block)
                    {
                        int num_splits = (n + max_n_block - 1) / max_n_block;
                        int even_block = (n + num_splits - 1) / num_splits;
                        even_block = (even_block + 63) / 64 * 64;
                        n_task_block = even_block;
                    }
                    else
                    {
                        n_task_block = calc_block;
                    }

                    int unroll = debugEnv().gemm.gemm_m_unroll_factor;
                    if (unroll != 1 && unroll != 2)
                        unroll = 2;

#pragma omp for collapse(2) schedule(static)
                    for (int n_task = 0; n_task < n; n_task += n_task_block)
                    {
                        for (int i = 0; i < m; i += unroll)
                        {
                            int rows_left = m - i;
                            int rows_to_process = (rows_left >= unroll) ? unroll : rows_left;

                            const Q8_1Block *blocks = all_blocks + i * k_blocks;

                            int n_end = std::min(n, n_task + n_task_block);
                            for (int n_blk = n_task; n_blk < n_end; n_blk += 64)
                            {
                                size_t weights_offset = (size_t)(n_blk / 64) * (k * 64);
                                const int8_t *b_ptr = pw.packed_data.data() + weights_offset;

                                const int32_t *comp_ptr = pw.compensation.data() + n_blk;
                                const float *scales_ptr = pw.scales.data() + n_blk;
                                const float *mins_ptr = pw.mins.data() + n_blk;

                                volatile QuantisedGemmParams params_v;
                                params_v.A = blocks;
                                params_v.B_packed = b_ptr;
                                params_v.comp = comp_ptr;
                                params_v.scales = scales_ptr;
                                params_v.mins = pw.has_mins ? mins_ptr : nullptr;
                                params_v.C = nullptr; // FP32 output not used
                                params_v.K_blocks = k_blocks;
                                params_v.N = 64;
                                params_v.ldc = N_padded;
                                params_v.bias = bias_ptr ? (bias_ptr + n_blk) : nullptr;
                                params_v.mask = nullptr;
                                params_v.A_stride = k_blocks * sizeof(Q8_1Block);
                                params_v.local_max = nullptr;
                                params_v.local_sum = nullptr;
                                params_v.do_softmax = false;
                                params_v.gate_input = nullptr;
                                params_v.do_swiglu = false;

                                // Q16_1 output parameters
                                params_v.output_format = GemmOutputFormat::Q16_1;
                                params_v.C_q8_1 = nullptr; // Not used for Q16_1
                                params_v.C_q8_1_stride = 0;

                                // Calculate Q16_1 output pointer
                                // Output block index = n_blk / q16_block_size
                                int block_index = n_blk / q16_block_size;
                                uint8_t *output_ptr = static_cast<uint8_t *>(C_q16_1) +
                                                      i * output_stride +
                                                      block_index * q16_block_bytes;

                                params_v.C_q16_1 = output_ptr;
                                params_v.C_q16_1_stride = static_cast<int>(output_stride);
                                params_v.q16_block_size = q16_block_size;

                                // Copy to non-volatile for kernel call
                                QuantisedGemmParams params;
                                std::memcpy(&params, const_cast<QuantisedGemmParams *>(&params_v), sizeof(params));

                                asm volatile("" ::: "memory");

                                if (rows_to_process == 2)
                                    kernel_m2(&params);
                                else
                                    kernel(&params);
                            }
                        }
                    }
                };

                OMP_WORKSHARE_REGION(do_gemm_q16_1_work);

                return true;
            }

            /**
             * @brief GEMM with FP32 input, outputting to Q8_1 format
             *
             * Quantizes FP32 activations to Q8_1, performs GEMM, then requantizes
             * the output to Q8_1. Combines quantize_activations() and
             * multiply_with_precomputed_q8_1_to_q8_1() in one call.
             *
             * @param A FP32 activation matrix [m, k]
             * @param C_q8_1 Output Q8_1 blocks [m, (n+31)/32] - must be pre-allocated
             * @param m Number of rows (sequence length)
             * @param n Output features (columns)
             * @param k Input features
             * @param mpi_ctx MPI context
             * @param device_idx Device index
             *
             * @return true on success
             */
            bool multiply_to_q8_1(
                const float *A,
                void *C_q8_1,
                int m, int n, int k,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                DeviceWorkspaceManager *workspace = nullptr) override
            {
                (void)workspace; // CPU kernel doesn't need external workspace
                // Quantize activations first
                int k_blocks = (k + 31) / 32;
                std::vector<uint8_t> q8_1_buffer(m * k_blocks * sizeof(Q8_1Block) + 64);

                // Use internal quantization
                quantize_activations(A, q8_1_buffer.data(), m, k);

                // Then use Q8_1→Q8_1 path (no bias for this simple interface)
                return multiply_with_precomputed_q8_1_to_q8_1(
                    q8_1_buffer.data(), C_q8_1, m, n, k,
                    nullptr, // bias
                    false, mpi_ctx, device_idx);
            }

            /**
             * @brief Tensor-based GEMM with full type-aware dispatch
             *
             * Handles all input/output tensor type combinations:
             *
             * **Input (A) dispatch**:
             * - Q8_1Tensor: Zero-copy, uses pre-quantized blocks directly
             * - IActivationTensor (FP32, BF16, FP16): Uses tensor's quantize_to_q8_1()
             * - Other tensors: Fallback to FP32 path via data()
             *
             * **Output (C) dispatch**:
             * - Q8_1Tensor: Output directly to Q8_1 blocks (avoids dequant→requant)
             * - FP32/BF16/FP16: Output to FP32 via mutable_data()
             *
             * @param A Input activations tensor [m, k]
             * @param C Output tensor [m, n] (Q8_1, FP32, BF16, or FP16)
             * @param transpose_B Whether B is transposed (ignored, weights pre-packed)
             * @param alpha Scale factor for result
             * @param beta Accumulate factor (0.0 = overwrite, 1.0 = add)
             * @param bias Optional bias tensor [n] to add after GEMM (nullptr = no bias)
             * @param mpi_ctx MPI context
             * @param device_idx Device index
             *
             * @return true on success
             */
            bool multiply_tensor(
                const TensorBase *A, TensorBase *C,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const TensorBase *bias = nullptr,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                DeviceWorkspaceManager *workspace = nullptr) override
            {
                (void)transpose_B; // Weights are pre-packed
                (void)workspace;   // CPU kernel doesn't need external workspace

                const auto &a_shape = A->shape();
                const auto &c_shape = C->shape();
                int m = static_cast<int>(a_shape[0]);
                int k = static_cast<int>(a_shape.size() > 1 ? a_shape[1] : 1);
                int n = static_cast<int>(c_shape.size() > 1 ? c_shape[1] : c_shape[0]);

                // Determine input/output types
                const TensorType a_type = A->native_type();
                const TensorType c_type = C->native_type();

                // =====================================================================
                // Q8_1 output paths (avoid dequant→requant roundtrip)
                // =====================================================================
                if (c_type == TensorType::Q8_1)
                {
                    auto *C_q8 = static_cast<Q8_1Tensor *>(C);

                    if (a_type == TensorType::Q8_1)
                    {
                        // Q8_1 → Q8_1: Optimal path, both quantized
                        auto *A_q8 = static_cast<const Q8_1Tensor *>(A);
                        return multiply_with_precomputed_q8_1_to_q8_1(
                            A_q8->typed_data(),
                            C_q8->mutable_typed_data(),
                            m, n, k,
                            bias,
                            false, // accumulate (not supported for Q8_1 output)
                            mpi_ctx, device_idx);
                    }
                    else
                    {
                        // FP32/BF16/FP16 → Q8_1: Quantize input on-the-fly, output to Q8_1
                        return multiply_to_q8_1(
                            A->data(), // Dequant to FP32 if needed
                            C_q8->mutable_typed_data(),
                            m, n, k,
                            mpi_ctx, device_idx);
                    }
                }

                // =====================================================================
                // FP32 output paths
                // =====================================================================
                if (a_type == TensorType::Q8_1)
                {
                    // Q8_1 → FP32: Use pre-quantized blocks, output FP32
                    return multiply_q8_1_direct(
                        static_cast<const Q8_1Tensor *>(A),
                        C->mutable_data(), m, n, k,
                        bias, beta != 0.0f, alpha, beta, mpi_ctx, device_idx);
                }

                // Any IActivationTensor → FP32: Use tensor's quantize_to_q8_1()
                const IActivationTensor *activation = dynamic_cast<const IActivationTensor *>(A);
                if (activation)
                {
                    // Allocate Q8_1 buffer for input quantization
                    size_t q8_1_size = IActivationTensor::get_q8_1_buffer_size(m, k);
                    std::vector<uint8_t> q8_1_buffer(q8_1_size + 64); // +64 for alignment

                    // Quantize using tensor's type-specific implementation
                    if (!activation->quantize_to_q8_1(q8_1_buffer.data(), m, k))
                    {
                        LOG_ERROR("quantize_to_q8_1 failed in multiply_tensor");
                        return false;
                    }

                    // Use pre-quantized path → FP32 output
                    return multiply_with_precomputed_q8_1(
                        q8_1_buffer.data(), C->mutable_data(), m, n, k,
                        bias,
                        beta != 0.0f, alpha, beta,
                        mpi_ctx, device_idx,
                        GemmFusedOps::none());
                }

                // Fallback: FP32 path with inline quantization
                return multiply(A->data(), C->mutable_data(), m, n, k, transpose_B, alpha, beta, mpi_ctx, device_idx);
            }

            /**
             * @brief Tensor-based GEMM with explicit dimensions
             *
             * This overload uses caller-provided m, n, k instead of inferring from tensor shapes.
             * Essential for:
             * - Pre-allocated buffers (tensor shape > actual data)
             * - Row-parallel GEMM (local n differs from tensor allocation)
             * - Processing subset of batched buffers
             *
             * @param A Input activations tensor [>=m, >=k]
             * @param C Output tensor [>=m, >=n]
             * @param m Number of rows to process
             * @param n Number of output columns
             * @param k Number of input columns
             * @param transpose_B Whether B is transposed (ignored, weights pre-packed)
             * @param alpha Scale factor
             * @param beta Accumulate factor
             * @param bias Optional bias tensor [n] to add after GEMM (nullptr = no bias)
             * @param mpi_ctx MPI context
             * @param device_idx Device index
             *
             * @return true on success
             */
            bool multiply_tensor(
                const TensorBase *A, TensorBase *C,
                int m, int n, int k,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const TensorBase *bias = nullptr,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                DeviceWorkspaceManager *workspace = nullptr) override
            {
                (void)transpose_B; // Weights are pre-packed
                (void)workspace;   // CPU kernel doesn't need external workspace

                // Determine input/output types
                const TensorType a_type = A->native_type();
                const TensorType c_type = C->native_type();

                // =====================================================================
                // Q8_1 output paths (avoid dequant→requant roundtrip)
                // =====================================================================
                if (c_type == TensorType::Q8_1)
                {
                    auto *C_q8 = static_cast<Q8_1Tensor *>(C);

                    if (a_type == TensorType::Q8_1)
                    {
                        // Q8_1 → Q8_1: Optimal path, both quantized
                        auto *A_q8 = static_cast<const Q8_1Tensor *>(A);
                        return multiply_with_precomputed_q8_1_to_q8_1(
                            A_q8->typed_data(),
                            C_q8->mutable_typed_data(),
                            m, n, k,
                            bias,
                            false, // accumulate (not supported for Q8_1 output)
                            mpi_ctx, device_idx);
                    }
                    else
                    {
                        // FP32/BF16/FP16 → Q8_1: Quantize input on-the-fly, output to Q8_1
                        return multiply_to_q8_1(
                            A->data(), // Dequant to FP32 if needed
                            C_q8->mutable_typed_data(),
                            m, n, k,
                            mpi_ctx, device_idx);
                    }
                }

                // =====================================================================
                // FP32 output paths
                // =====================================================================
                if (a_type == TensorType::Q8_1)
                {
                    // Q8_1 → FP32: Use pre-quantized blocks, output FP32
                    return multiply_q8_1_direct(
                        static_cast<const Q8_1Tensor *>(A),
                        C->mutable_data(), m, n, k,
                        bias, beta != 0.0f, alpha, beta, mpi_ctx, device_idx);
                }

                // Any IActivationTensor → FP32: Use tensor's quantize_to_q8_1()
                const IActivationTensor *activation = dynamic_cast<const IActivationTensor *>(A);
                if (activation)
                {
                    // Allocate Q8_1 buffer for input quantization
                    size_t q8_1_size = IActivationTensor::get_q8_1_buffer_size(m, k);
                    std::vector<uint8_t> q8_1_buffer(q8_1_size + 64); // +64 for alignment

                    // Quantize using tensor's type-specific implementation
                    if (!activation->quantize_to_q8_1(q8_1_buffer.data(), m, k))
                    {
                        LOG_ERROR("quantize_to_q8_1 failed in multiply_tensor (explicit dims)");
                        return false;
                    }

                    // Use pre-quantized path → FP32 output
                    return multiply_with_precomputed_q8_1(
                        q8_1_buffer.data(), C->mutable_data(), m, n, k,
                        bias,
                        beta != 0.0f, alpha, beta,
                        mpi_ctx, device_idx,
                        GemmFusedOps::none());
                }

                // Fallback: FP32 path with inline quantization
                return multiply(A->data(), C->mutable_data(), m, n, k, transpose_B, alpha, beta, mpi_ctx, device_idx);
            }

        private:
            bool compute_q8_1_strided_gemm_softmax(
                const void *A, const void *B, float *C,
                int m, int n, int k,
                int stride_q_bytes, int stride_k_bytes,
                float scale, const float *mask, bool is_causal, int causal_offset)
            {
                static QuantisedGemmJit_Q8_1_OnlineSoftmax kernel_m4(4);
                static QuantisedGemmJit_Q8_1_OnlineSoftmax kernel_m1(1);

                int m_blocking = 4;

                // Nested-safe parallel execution over M blocks
                auto softmax_gemm_work = [&]()
                {
#pragma omp for schedule(dynamic)
                    for (int i = 0; i < m; i += m_blocking)
                    {
                        int current_m = std::min(m_blocking, m - i);

                        OnlineSoftmaxParams params;
                        params.Q = static_cast<const char *>(A) + i * stride_q_bytes;
                        params.K = B;
                        params.C = C + i * n;
                        params.M = current_m;
                        params.N = n;
                        params.K_blocks = (k + 31) / 32;
                        params.Q_stride_bytes = stride_q_bytes;
                        params.K_stride_bytes = stride_k_bytes;
                        params.C_stride_bytes = n * sizeof(float);
                        params.scale = scale;

                        if (mask)
                        {
                            params.mask = mask + i * n;
                            params.mask_stride_bytes = n * sizeof(float);
                        }
                        else
                        {
                            params.mask = nullptr;
                            params.mask_stride_bytes = 0;
                        }

                        if (current_m == 4)
                        {
                            auto func = kernel_m4.get_kernel();
                            func(&params);
                        }
                        else
                        {
                            // Handle tail with M1 kernel loop
                            auto func = kernel_m1.get_kernel();
                            for (int j = 0; j < current_m; ++j)
                            {
                                OnlineSoftmaxParams p = params;
                                p.Q = static_cast<const char *>(params.Q) + j * stride_q_bytes;
                                p.C = static_cast<float *>(params.C) + j * n;
                                if (mask)
                                    p.mask = static_cast<const float *>(params.mask) + j * n;
                                func(&p);
                            }
                        }
                    }
                };
                OMP_WORKSHARE_REGION(softmax_gemm_work);
                return true;
            }

            /**
             * @brief Direct Q8_1 activation path (no float conversion)
             *
             * Uses pre-quantized blocks directly from Q8_1Tensor.
             * Eliminates the float→Q8_1 conversion overhead.
             */
            bool multiply_q8_1_direct(
                const Q8_1Tensor *A_tensor, float *C,
                int m, int n, int k,
                const TensorBase *bias,
                bool accumulate, float alpha, float beta,
                const MPIContext *ctx, int device_idx)
            {
                (void)ctx;
                (void)device_idx;
                (void)alpha; // Currently ignored (assumed 1.0)

                const float *bias_ptr = bias ? bias->fp32_data() : nullptr;

                // Validate Q8_1Tensor before processing
                if (!A_tensor)
                {
                    LOG_ERROR("[multiply_q8_1_direct] A_tensor is null");
                    return false;
                }

                // Check tensor dimensions match expected
                const auto &shape = A_tensor->shape();
                if (shape.size() != 2)
                {
                    LOG_ERROR("[multiply_q8_1_direct] A_tensor has " << shape.size() << " dims, expected 2");
                    return false;
                }
                if (static_cast<int>(shape[0]) < m || static_cast<int>(shape[1]) < k)
                {
                    LOG_ERROR("[multiply_q8_1_direct] A_tensor shape [" << shape[0] << "," << shape[1]
                                                                        << "] too small for m=" << m << " k=" << k);
                    return false;
                }

                // Validate Q8_1 blocks are accessible
                const size_t expected_blocks = ((m * ((k + 31) / 32)));
                const void *raw_ptr = A_tensor->get_raw_block_at(0, 0);
                if (!raw_ptr)
                {
                    LOG_ERROR("[multiply_q8_1_direct] get_raw_block_at(0,0) returned null");
                    return false;
                }

                LOG_TRACE("[multiply_q8_1_direct] Q8_1 input validated: m=" << m << " n=" << n << " k=" << k
                                                                            << " shape=[" << shape[0] << "," << shape[1] << "]"
                                                                            << " expected_blocks=" << expected_blocks
                                                                            << " raw_ptr=" << raw_ptr);

                // Get reference to packed weights (either external or owned)
                const auto &pw = packed_weights();

                // Check dimensions
                if (n != pw.N || k != pw.K)
                {
                    LOG_ERROR("Dimension mismatch in QuantisedGemmKernel::multiply_q8_1_direct: "
                              << "n=" << n << " pw.N=" << pw.N << " k=" << k << " pw.K=" << pw.K);
                    return false;
                }

                // Validate packed weights data is allocated
                if (pw.packed_data.empty())
                {
                    LOG_ERROR("[multiply_q8_1_direct] packed_data is empty! n=" << n << " k=" << k);
                    return false;
                }

                // Calculate expected sizes for validation
                int K_blocks = (k + 31) / 32;
                int K_padded = K_blocks * 32;
                int N_padded = (n + 63) / 64 * 64;
                size_t expected_packed_size = (size_t)K_padded * N_padded;

                if (pw.packed_data.size() < expected_packed_size)
                {
                    LOG_ERROR("[multiply_q8_1_direct] packed_data too small: " << pw.packed_data.size()
                                                                               << " < expected " << expected_packed_size
                                                                               << " (n=" << n << " k=" << k << " N_padded=" << N_padded << " K_padded=" << K_padded << ")");
                    return false;
                }

                LOG_TRACE("[multiply_q8_1_direct] packed_data validated: size=" << pw.packed_data.size()
                                                                                << " expected=" << expected_packed_size << " has_mins=" << pw.has_mins);

                // Get JIT kernels
                static QuantisedGemmJit_M1 jit;
                static QuantisedGemmJit_M2 jit_m2;
                auto kernel = jit.get_kernel();
                auto kernel_m2 = jit_m2.get_kernel();

                // Validate JIT kernel function pointers
                if (!kernel)
                {
                    LOG_ERROR("[multiply_q8_1_direct] JIT M1 kernel is null!");
                    return false;
                }
                if (!kernel_m2)
                {
                    LOG_ERROR("[multiply_q8_1_direct] JIT M2 kernel is null!");
                    return false;
                }

                // Handle beta scaling / zeroing (nested-safe)
                if (!accumulate || beta == 0.0f)
                {
                    auto zero_work = [&]()
                    {
#pragma omp for schedule(static)
                        for (size_t i = 0; i < (size_t)m * n; ++i)
                            C[i] = 0.0f;
                    };
                    OMP_WORKSHARE_REGION(zero_work);
                }
                else if (beta != 1.0f)
                {
                    auto scale_work = [&]()
                    {
#pragma omp for schedule(static)
                        for (size_t i = 0; i < (size_t)m * n; ++i)
                            C[i] *= beta;
                    };
                    OMP_WORKSHARE_REGION(scale_work);
                }

                int k_blocks = k / 32;
                int blocks_per_row_output = (n + 63) / 64;
                // N_padded already defined above for validation

                // Get direct access to Q8_1 blocks (zero-copy!)
                // A_tensor stores blocks in row-major order: [m, k_blocks] of Q8_1Block
                const size_t blocks_per_row_input = (k + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;

#pragma omp parallel
                {
                    // Detect cache sizes (same as FP32 path)
                    int num_threads = omp_get_num_threads();
                    long long l2_size = cpu_l2_cache_size();
                    if (l2_size == 0)
                        l2_size = 1024 * 1024;

                    long long l3_size = cpu_l3_cache_size();
                    if (l3_size == 0)
                        l3_size = l2_size * num_threads;

                    long long l2_limit = (long long)(l2_size * 0.9);
                    long long l3_limit_per_thread = (long long)(l3_size * 0.9 / num_threads);
                    long long block_size_limit = std::min(l2_limit, l3_limit_per_thread);
                    if (block_size_limit < 65536)
                        block_size_limit = 65536;

                    int max_n_block = block_size_limit / k;
                    max_n_block = (max_n_block / 64) * 64;
                    if (max_n_block < 64)
                        max_n_block = 64;

                    int target_tasks = num_threads * 4;
                    int m_tasks = (m + 1) / 2;
                    if (m_tasks < 1)
                        m_tasks = 1;

                    int needed_n_tasks = (target_tasks + m_tasks - 1) / m_tasks;
                    if (needed_n_tasks < 1)
                        needed_n_tasks = 1;

                    int calc_block = (n + needed_n_tasks - 1) / needed_n_tasks;
                    calc_block = (calc_block + 63) / 64 * 64;
                    if (calc_block < 64)
                        calc_block = 64;

                    int n_task_block;
                    if (calc_block > max_n_block)
                    {
                        int num_splits = (n + max_n_block - 1) / max_n_block;
                        int even_block = (n + num_splits - 1) / num_splits;
                        even_block = (even_block + 63) / 64 * 64;
                        n_task_block = even_block;
                    }
                    else
                    {
                        n_task_block = calc_block;
                    }

                    // Standard path: Collapse M and N for maximum parallelism
                    // (K-tiling path would need similar adaptation)
                    int unroll = debugEnv().gemm.gemm_m_unroll_factor;
                    if (unroll != 1 && unroll != 2)
                        unroll = 2;

#pragma omp for collapse(2) schedule(static)
                    for (int n_task = 0; n_task < n; n_task += n_task_block)
                    {
                        for (int i = 0; i < m; i += unroll)
                        {
                            int rows_left = m - i;
                            int rows_to_process = (rows_left >= unroll) ? unroll : rows_left;

                            // Get pointer to first block for row i
                            // Use get_raw_block_at to get the Q8_1Block* directly
                            const Q8_1Block *blocks = static_cast<const Q8_1Block *>(
                                A_tensor->get_raw_block_at(i, 0));

                            int n_end = std::min(n, n_task + n_task_block);
                            for (int n_blk = n_task; n_blk < n_end; n_blk += 64)
                            {
                                size_t weights_offset = (size_t)(n_blk / 64) * (k * 64);
                                const int8_t *b_ptr = pw.packed_data.data() + weights_offset;

                                const int32_t *comp_ptr = pw.compensation.data() + n_blk;
                                const float *scales_ptr = pw.scales.data() + n_blk;
                                const float *mins_ptr = pw.mins.data() + n_blk;

                                // Use volatile to prevent aggressive optimization of params struct
                                volatile QuantisedGemmParams params_v;
                                params_v.A = blocks;
                                params_v.B_packed = b_ptr;
                                params_v.comp = comp_ptr;
                                params_v.scales = scales_ptr;
                                params_v.mins = pw.has_mins ? mins_ptr : nullptr;
                                params_v.C = C + i * n + n_blk;
                                params_v.K_blocks = k_blocks;
                                params_v.N = 64;
                                params_v.ldc = N_padded; // Stride for comp/scales arrays, NOT C output
                                params_v.bias = bias_ptr ? bias_ptr + n_blk : nullptr;
                                params_v.mask = nullptr;
                                params_v.A_stride = blocks_per_row_input * sizeof(Q8_1Block);
                                params_v.local_max = nullptr;
                                params_v.local_sum = nullptr;
                                params_v.do_softmax = false;

                                // Copy to non-volatile for kernel call
                                QuantisedGemmParams params;
                                std::memcpy(&params, const_cast<QuantisedGemmParams *>(&params_v), sizeof(params));

                                // Memory barrier to prevent reordering (workaround for optimization issue)
                                asm volatile("" ::: "memory");

                                if (rows_to_process == 2)
                                    kernel_m2(&params);
                                else
                                    kernel(&params);
                            }
                        }
                    }
                }

                return true;
            }

            // Packed weights - owned by this kernel (used when external_packed_ is null)
            QuantisedPackedWeights packed_weights_;

            // External packed weights - pointer to data owned by tensor's cache_
            // When non-null, this kernel references external data instead of packed_weights_
            const QuantisedPackedWeights *external_packed_ = nullptr;
        };

    }
}
