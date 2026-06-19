/**
 * @file TensorKernels.h
 * @brief Kernel interfaces for tensor operations
 *
 * All kernels accept IMPIContext (distributed coordination) and device_idx (execution device).
 * Kernels are created by tensors via createGemm(), createRoPE(), etc.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../utils/MPIContext.h"
#include "../interfaces/IWorkspaceConsumer.h"
#include "../kernels/IPackedWeights.h"
#include "BlockStructures.h"
#include "KernelSnapshotInfo.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace llaminar2
{
    // Forward declarations
    class ITensor; // Device-agnostic tensor interface
    class TensorBase;
    struct PreparedEmbeddingHandle;
    struct Q8_1Block;
    class IDeviceContext; // For kernel execute() interface

    /**
     * @brief Device-readable native-VNNI matrix descriptor.
     *
     * This intentionally carries only raw device pointers and compact format
     * metadata so MoE grouped kernels can select expert weights by descriptor
     * without depending on backend-specific GEMM classes.
     */
    struct DeviceNativeVNNIMatrixDesc
    {
        const uint8_t *payload = nullptr;
        const void *scales = nullptr;
        const void *mins = nullptr;
        const void *emins = nullptr;
        int n = 0;
        int k = 0;
        uint32_t blocks_per_row = 0;
        uint8_t codebook_id = 0;
        uint8_t reserved[3] = {0, 0, 0};

        bool valid() const
        {
            return payload && scales && n > 0 && k > 0 && blocks_per_row > 0;
        }
    };

    // =============================================================================
    // Fused Operation Configuration
    // =============================================================================

    /**
     * @brief Fused operation types that can be applied after GEMM
     *
     * These operations are fused into the GEMM kernel to avoid extra memory passes.
     */
    enum class GemmFusedOpType : uint8_t
    {
        None = 0, ///< No post-GEMM fusion
        SwiGLU,   ///< SwiGLU activation: output *= swish(gate) = output * gate * sigmoid(gate)
        Softmax,  ///< Row-wise softmax: output = softmax(output + mask)
        ReLU,     ///< ReLU activation: output = max(0, output)
        GeLU,     ///< GeLU activation: output = output * 0.5 * (1 + erf(output / sqrt(2)))
        SiLU      ///< SiLU/Swish activation: output = output * sigmoid(output)
    };

    /**
     * @brief Configuration for fused operations after GEMM
     *
     * This struct consolidates all fused operation parameters into a single
     * extensible configuration object. Use the static factory methods for
     * convenient construction.
     *
     * Design rationale:
     * - Avoids interface bloat from separate methods for each fused op
     * - Extensible: add new fused ops without breaking API
     * - Zero-cost when unused (default-constructed = no fusion)
     *
     * Usage:
     * @code
     * // No fusion (default)
     * kernel->multiply_with_precomputed_q8_1(..., GemmFusedOps::none());
     *
     * // SwiGLU fusion
     * kernel->multiply_with_precomputed_q8_1(..., GemmFusedOps::swiglu(gate_output));
     *
     * // Softmax fusion (for attention)
     * kernel->multiply_with_precomputed_q8_1(..., GemmFusedOps::softmax(1.0f / sqrt(d_k), mask));
     * @endcode
     */
    struct GemmFusedOps
    {
        GemmFusedOpType op_type = GemmFusedOpType::None;

        // --- SwiGLU parameters ---
        const float *gate_input = nullptr; ///< Gate tensor for SwiGLU [m, n] (output *= swish(gate))

        // --- Softmax parameters ---
        float softmax_scale = 1.0f;          ///< Pre-softmax scale (typically 1/sqrt(d_k))
        const float *softmax_mask = nullptr; ///< Additive mask before softmax [m, n] or nullptr
        bool is_causal = false;              ///< Apply causal (lower-triangular) mask
        int softmax_axis = 1;                ///< Axis for softmax reduction (typically 1 = row-wise)

        // --- Online softmax outputs (for flash attention) ---
        float *online_max = nullptr; ///< Output: per-row max values [m]
        float *online_sum = nullptr; ///< Output: per-row sum of exp values [m]

        // =========================================================================
        // Factory methods for convenient construction
        // =========================================================================

        /**
         * @brief No fused operation (default)
         */
        static GemmFusedOps none()
        {
            return GemmFusedOps{};
        }

        /**
         * @brief SwiGLU fusion: output = output * swish(gate) = output * gate * sigmoid(gate)
         *
         * Used in FFN: up_output = up_proj * swish(gate_proj)
         *
         * @param gate Gate tensor [m, n] (must match output dimensions)
         */
        static GemmFusedOps swiglu(const float *gate)
        {
            GemmFusedOps ops;
            ops.op_type = GemmFusedOpType::SwiGLU;
            ops.gate_input = gate;
            return ops;
        }

        /**
         * @brief Softmax fusion: output = softmax(output * scale + mask)
         *
         * Used in attention: scores = softmax(Q @ K^T / sqrt(d_k) + mask)
         *
         * @param scale Pre-softmax scale (typically 1/sqrt(d_k))
         * @param mask Optional additive mask [m, n] (nullptr = no mask)
         * @param causal Apply causal (lower-triangular) mask
         */
        static GemmFusedOps softmax(float scale = 1.0f, const float *mask = nullptr, bool causal = false)
        {
            GemmFusedOps ops;
            ops.op_type = GemmFusedOpType::Softmax;
            ops.softmax_scale = scale;
            ops.softmax_mask = mask;
            ops.is_causal = causal;
            return ops;
        }

        /**
         * @brief Online softmax for flash attention (returns max/sum for rescaling)
         *
         * @param scale Pre-softmax scale
         * @param out_max Output buffer for per-row max [m]
         * @param out_sum Output buffer for per-row exp sum [m]
         * @param mask Optional additive mask
         * @param causal Apply causal mask
         */
        static GemmFusedOps online_softmax(float scale, float *out_max, float *out_sum,
                                           const float *mask = nullptr, bool causal = false)
        {
            GemmFusedOps ops;
            ops.op_type = GemmFusedOpType::Softmax;
            ops.softmax_scale = scale;
            ops.softmax_mask = mask;
            ops.is_causal = causal;
            ops.online_max = out_max;
            ops.online_sum = out_sum;
            return ops;
        }

        /**
         * @brief Check if any fused operation is enabled
         */
        bool has_fusion() const { return op_type != GemmFusedOpType::None; }

        /**
         * @brief Check if specific operation type is enabled
         */
        bool is_swiglu() const { return op_type == GemmFusedOpType::SwiGLU; }
        bool is_softmax() const { return op_type == GemmFusedOpType::Softmax; }
    };

    // =============================================================================
    // Activation Format Enumeration
    // =============================================================================

    /**
     * @brief Activation precision format enumeration
     *
     * Used by ITensorGemm to dispatch to appropriate GEMM implementation
     * based on activation and weight precision.
     *
     * Format semantics:
     * - FP32: 32-bit floating point (IEEE 754 binary32)
     * - BF16: Brain Float 16 (truncated FP32, 8-bit exponent)
     * - FP16: Half precision (IEEE 754 binary16)
     * - INT8: 8-bit integer (quantized activation, needs INT32 accumulator)
     * - Q8_0: Quantized 8-bit with per-block scales (weight format, not activation)
     * - Q8_1: Quantized 8-bit with per-block scales + zero points (activation format)
     */
    enum class ActivationFormat
    {
        FP32, ///< 32-bit float (native CPU, universal fallback)
        BF16, ///< Brain Float 16 (AVX512_BF16, OneDNN native)
        FP16, ///< Half precision (FP16 instructions, OneDNN native)
        INT8, ///< Quantized 8-bit integer (AVX512-VNNI, INT32 accumulator)
        Q8_0, ///< Quantized 8-bit weight format (block-scaled)
        Q8_1  ///< Quantized 8-bit activation format (block-scaled with zero-point)
    };

    /**
     * @brief Block decoder strategy interface for quantized tensors (FP32 decode path)
     *
     * Provides format-specific dequantization for block-quantized formats.
     * Implementations MUST be header-only with always_inline to ensure zero overhead
     * when called from GEMM hot paths.
     *
     * Supported formats: IQ4_NL, Q6_K, Q8_0, etc.
     *
     * NOTE: This interface decodes to FP32. For integer GEMM kernels that want
     * native int-domain access, use IINT8Unpackable instead.
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
     * @brief Base kernel interface
     *
     * All kernels inherit from ITensorKernel, which in turn inherits from
     * IKernelSnapshotCapable. This enforces that all kernel implementations
     * MUST implement getKernelSnapshotInfo() for snapshot support.
     *
     * @note The pure virtual getKernelSnapshotInfo() method is inherited
     *       from IKernelSnapshotCapable. Subclasses MUST override it or
     *       the code will not compile.
     */
    class ITensorKernel : public IKernelSnapshotCapable
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

        /**
         * @brief Set the GPU stream for kernel dispatch (GPU graph capture support)
         *
         * When non-null, GPU kernels dispatch work on this stream instead of the
         * default stream. This is required for GPU graph capture — all kernels must
         * dispatch to the capture stream for nodes to be recorded in the graph.
         *
         * CPU kernels ignore this (default no-op).
         *
         * @param stream Opaque stream pointer (hipStream_t or cudaStream_t cast to void*)
         */
        virtual void setGPUStream(void *stream) { (void)stream; }

        // =====================================================================
        // Session Lifecycle (kernel state that depends on input data, not weights)
        // =====================================================================

        /**
         * @brief Reset any input-dependent cached state (session boundary)
         *
         * Called by the orchestrator when starting a new inference session
         * (after clear_cache). Kernels MUST reset any state that depends on
         * input data (token IDs, sequence lengths, position offsets).
         * Weight-dependent state (packed weights, GPU handles) is preserved.
         *
         * Default is no-op for kernels without dynamic state.
         */
        virtual void resetDynamicState() {}

        /**
         * @brief Check if this kernel has active input-dependent cached state
         *
         * Used by debug-mode assertions to verify no kernel retains stale
         * dynamic state after a session reset.
         *
         * @return true if the kernel has cached input-dependent data
         */
        virtual bool hasDynamicStateActive() const { return false; }
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
    // Forward declaration for workspace
    class DeviceWorkspaceManager;

    class ITensorGemm : public ITensorKernel
    {
    public:
        /**
         * @brief Tensor-based matrix multiplication: C = alpha * A @ B + beta * C
         *
         * This interface accepts tensors directly, enabling:
         * - **Type introspection**: Kernel can detect activation format (FP32, Q8_1, BF16, etc.)
         * - **Zero-copy quantized path**: If A is Q8_1Tensor, skip float→Q8_1 conversion
         * - **Flexible output**: C can be any precision (FP32, BF16, etc.)
         *
         * **Dispatch logic** (typical implementation):
         * 1. Check A->native_type() to determine activation format
         * 2. If Q8_1: Use quantized blocks directly (via IQ8_1Decodable or block access)
         * 3. If FP32: Use FP32 GEMM path directly
         * 4. If BF16: Convert or use native BF16 GEMM if available
         *
         * @param A Input activations tensor [m, k] (any format)
         * @param C Output tensor [m, n] (must be mutable, typically FP32)
         * @param transpose_B Whether B (packed weights) is transposed (typical: true)
         * @param alpha Scale factor for A@B
         * @param beta Scale factor for existing C (for fused add)
         * @param bias Optional bias tensor [n] to add after GEMM (nullptr = no bias)
         * @param mpi_ctx MPI context for distributed execution
         * @param device_idx Device index for execution
         * @param workspace Optional pre-allocated workspace (nullptr = kernel allocates)
         * @param activation_row_offset Row offset into A to start reading from (default 0).
         *        Used by LMHeadStage to compute only the last token's logits during prefill,
         *        avoiding seq_len× redundant GEMM compute and D2H copy.
         *
         * @return true on success, false if format combination not supported
         *
         * @note Default implementation returns false (not supported).
         *       Kernels that include Tensors.h can override with optimized paths.
         */
        virtual bool multiply_tensor(
            const TensorBase *A, TensorBase *C,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const TensorBase *bias = nullptr,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            DeviceWorkspaceManager *workspace = nullptr,
            int activation_row_offset = 0)
        {
            (void)A;
            (void)C;
            (void)transpose_B;
            (void)alpha;
            (void)beta;
            (void)bias;
            (void)mpi_ctx;
            (void)device_idx;
            (void)workspace;
            (void)activation_row_offset;
            return false;
        }

        /**
         * @brief Tensor-based matrix multiplication with explicit dimensions
         *
         * This overload accepts explicit m, n, k dimensions, essential for:
         * - **Pre-allocated buffers**: Tensor shapes may be larger than actual data
         * - **Row-parallel GEMM**: Local output size differs from tensor allocation
         * - **Batched execution**: Processing subset of pre-allocated batch buffer
         *
         * @param A Input activations tensor [>=m, >=k] (actual data in first m rows)
         * @param C Output tensor [>=m, >=n] (writes to first m rows)
         * @param m Number of rows to process (may be < A->shape()[0])
         * @param n Number of output columns (may be < C->shape()[1])
         * @param k Number of input columns (may be < A->shape()[1])
         * @param transpose_B Whether B (packed weights) is transposed (typical: true)
         * @param alpha Scale factor for A@B
         * @param beta Scale factor for existing C (for fused add)
         * @param bias Optional bias tensor [n] to add after GEMM (nullptr = no bias)
         * @param mpi_ctx MPI context for distributed execution
         * @param device_idx Device index for execution
         * @param workspace Optional pre-allocated workspace (nullptr = kernel allocates)
         * @param activation_row_offset Row offset into A to start reading from (default 0).
         *        When non-zero, reads m rows starting at A[activation_row_offset, :].
         *
         * @return true on success, false if format combination not supported
         *
         * @note This is the PRIMARY GEMM interface. All implementations must provide this.
         *       This method:
         *       - Carries tensor type information for automatic format dispatch
         *       - Avoids parameter ordering ambiguity (transpose_B vs alpha vs beta)
         *       - Enables device coherence management
         *       - Supports shape validation and type checking
         */
        virtual bool multiply_tensor(
            const TensorBase *A, TensorBase *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const TensorBase *bias = nullptr,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            DeviceWorkspaceManager *workspace = nullptr,
            int activation_row_offset = 0) = 0;

        /**
         * @brief Fused SwiGLU activation + GEMM: output = (silu(gate) * up) @ B
         *
         * Eliminates the separate SwiGLU kernel and intermediate FP32 write/read by
         * fusing the activation function with the subsequent quantization and GEMM.
         * On GPU: silu(gate)*up is computed and quantized in a single kernel launch.
         *
         * @return true if fused path executed, false to fall back to separate SwiGLU + GEMM
         */
        virtual bool multiply_tensor_with_fused_swiglu(
            const TensorBase *gate,
            const TensorBase *up,
            TensorBase *output,
            int m, int n, int k,
            float alpha = 1.0f, float beta = 0.0f,
            DeviceWorkspaceManager *workspace = nullptr)
        {
            (void)gate;
            (void)up;
            (void)output;
            (void)m;
            (void)n;
            (void)k;
            (void)alpha;
            (void)beta;
            (void)workspace;
            return false; // Default: not supported, fall back to separate path
        }

        /**
         * @brief Grouped verifier-row SwiGLU + down projection.
         *
         * The MTP verifier publishes accepted rows from a compact M=2..4 graph,
         * but the published row must be numerically equivalent to serial decode.
         * Implementations that override this method must compute all verifier
         * rows through a grouped/concurrent path while preserving the M=1 decode
         * SwiGLU and GEMV contracts. Returning false is a hard capability miss:
         * stages must not hide an unsupported backend behind row-wise replay.
         *
         * @param gate Gate projection rows [m, k].
         * @param up Up projection rows [m, k].
         * @param output Down projection output [m, n].
         * @param m Verifier row count. Production MTP uses 2..4.
         * @param n Output width.
         * @param k Intermediate width.
         */
        virtual bool multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
            const TensorBase *gate,
            const TensorBase *up,
            TensorBase *output,
            int m, int n, int k,
            float alpha = 1.0f, float beta = 0.0f,
            DeviceWorkspaceManager *workspace = nullptr)
        {
            (void)gate;
            (void)up;
            (void)output;
            (void)m;
            (void)n;
            (void)k;
            (void)alpha;
            (void)beta;
            (void)workspace;
            return false;
        }

        /**
         * @brief Check if this kernel supports optimized fused multi-projection
         *
         * @return true if multiply_fused_tensor() has optimized implementation beyond sequential GEMMs
         */
        virtual bool supports_fused_projection() const
        {
            return false; // Default: no optimized fusion, uses sequential fallback
        }

        /**
         * @brief Prepare weights for efficient execution
         *
         * For CPU kernels: no-op (packing happens lazily in KernelFactory)
         * For GPU kernels: converts to INT8 + uploads to device memory
         *
         * Call this during weight preloading to avoid first-use overhead.
         * This is called by WeightManager::packGemmWeights() during preloading.
         */
        virtual void prepareWeights() {}

        /**
         * @brief Check whether weights have been successfully converted and uploaded.
         *
         * Returns true once prepareWeights() has completed everything it needs to do
         * (e.g. hipMemcpy/cudaMemcpy of VNNI data to device).  Callers can use this
         * to detect silent upload failures (prepareWeights() is void).
         */
        virtual bool weights_converted() const { return true; }

        /**
         * @brief Export a device-readable native-VNNI weight descriptor.
         *
         * Backends that store weights in native-VNNI form override this for
         * grouped MoE decode kernels. Unsupported formats return false so the
         * caller can use the existing sequential fallback.
         */
        virtual bool exportNativeVNNIMatrixDesc(DeviceNativeVNNIMatrixDesc &out)
        {
            out = {};
            return false;
        }

        // =====================================================================
        // Tensor-aware fused projection API (preferred for GPU execution)
        // =====================================================================

        /**
         * @brief Descriptor for tensor-aware fused multi-projection GEMM
         *
         * Unlike FusedProjectionDesc which uses raw pointers, this uses TensorBase
         * objects which manage their own device placement. The kernel handles
         * all device synchronization internally.
         */
        struct TensorProjectionDesc
        {
            ITensorGemm *kernel;    ///< GEMM kernel for this projection (with packed weights)
            TensorBase *output;     ///< Output tensor [m, n] - kernel manages device placement
            int n;                  ///< Output dimension (columns)
            const TensorBase *bias; ///< Optional bias tensor [n] (nullptr = no bias)
            const char *name;       ///< Name for debug logging

            TensorProjectionDesc(ITensorGemm *k, TensorBase *out, int n_dim,
                                 const TensorBase *b = nullptr,
                                 const char *nm = nullptr)
                : kernel(k), output(out), n(n_dim), bias(b), name(nm) {}
        };

        /**
         * @brief Tensor-aware fused multi-projection GEMM
         *
         * This is the preferred API for GPU execution. Unlike multiply_fused() which
         * takes raw pointers, this takes ITensor objects which manage their own device
         * placement. The kernel handles all device synchronization internally.
         *
         * **Device Handling**:
         * - Input tensor: ensureOnDevice() called if needed
         * - Output tensors: ensureOnDevice() called, transitionTo(DEVICE_AUTHORITATIVE) after write
         * - Weight tensors: managed by the kernel (already packed/uploaded)
         *
         * **For CPU execution**: Falls back to host pointers transparently
         * **For GPU execution**: Uses device pointers, no manual sync needed by caller
         *
         * @param input Input tensor [m, k] (FP32)
         * @param projections Vector of tensor projection descriptors
         * @param m Number of rows (batch_size * seq_len)
         * @param k Input dimension (must match all kernels' K dimension)
         * @param mpi_ctx MPI context for distributed execution
         * @param workspace Optional pre-allocated workspace (nullptr = kernel allocates)
         *
         * @return true on success, false on error
         */
        virtual bool multiply_fused_tensor(
            const TensorBase *input,
            const std::vector<TensorProjectionDesc> &projections,
            int m, int k,
            const IMPIContext *mpi_ctx = nullptr,
            DeviceWorkspaceManager *workspace = nullptr)
        {
            // Default implementation: call multiply_tensor() for each projection
            for (const auto &proj : projections)
            {
                if (!proj.kernel || !proj.output)
                {
                    return false; // Invalid projection
                }

                // Use tensor-aware multiply - kernel handles device placement
                // Note: Must pass transpose_B explicitly before alpha/beta to match signature:
                //   multiply_tensor(A, C, m, n, k, transpose_B, alpha, beta, bias, mpi_ctx, device_idx, workspace)
                bool success = proj.kernel->multiply_tensor(
                    input, proj.output,
                    m, proj.n, k,
                    true,      // transpose_B (weights are [K,N] stored as [N,K] transposed)
                    1.0f,      // alpha
                    0.0f,      // beta
                    proj.bias, // bias tensor (may be nullptr)
                    mpi_ctx,
                    -1, // device_idx (use default)
                    workspace);

                if (!success)
                {
                    return false;
                }
            }
            return true;
        }

        /**
         * @brief Grouped verifier-row projection with serial decode row math.
         *
         * MTP verifier graphs often evaluate M=2..4 candidate rows together.
         * A normal GEMM kernel may legally change accumulation order across
         * rows, K tiles, or projection groups, but verifier rows that publish
         * recurrent/KV state need the same per-row numerical contract as M=1
         * serial decode. Implementations that override this method must:
         *
         * - process all requested rows in one grouped/concurrent kernel path;
         * - use the same activation quantization and per-row accumulation order
         *   as their M=1 decode GEMV path;
         * - write every projection output in the supplied batched tensors; and
         * - return false rather than silently falling back to row-wise stage
         *   replay when that contract is unsupported.
         *
         * The default implementation is unsupported. Stages use this as the
         * Phase 9.8 promotion gate before retiring older serial verifier helpers.
         */
        virtual bool multiply_fused_verifier_rows_decode_equivalent(
            const TensorBase *input,
            const std::vector<TensorProjectionDesc> &projections,
            int m, int k,
            const IMPIContext *mpi_ctx = nullptr,
            DeviceWorkspaceManager *workspace = nullptr)
        {
            (void)input;
            (void)projections;
            (void)m;
            (void)k;
            (void)mpi_ctx;
            (void)workspace;
            return false;
        }

        /**
         * @brief Descriptor for fused multi-input down projections.
         *
         * Each descriptor has its own FP32 input (e.g., SwiGLU output per MoE expert),
         * unlike TensorProjectionDesc where all projections share one input.
         */
        struct FusedExpertDownDesc
        {
            ITensorGemm *kernel; ///< Down projection GEMM engine (with packed weights)
            const float *input;  ///< Per-expert FP32 input [k]
            float *output;       ///< Per-expert FP32 output [n]
            int n;               ///< Output dimension (d_model)
        };

        /**
         * @brief Fused multi-input GEMV for MoE expert down projections.
         *
         * Processes multiple (input, weight, output) triples in a single OMP region,
         * saving fork/join overhead (3×~8µs for 4 experts) and improving load balance
         * via nowait between projections (128 total chunks vs 4×32 with 28 threads).
         *
         * @param descs Array of expert down projection descriptors
         * @param num_descs Number of experts
         * @param m Always 1 for decode (single token)
         * @param k Input dimension (expert intermediate size)
         * @return true if fused path executed, false to fall back to sequential
         */
        virtual bool multiply_fused_expert_down(
            const FusedExpertDownDesc *descs, int num_descs,
            int m, int k)
        {
            (void)descs;
            (void)num_descs;
            (void)m;
            (void)k;
            return false;
        }

        /**
         * @brief GEMM with pre-quantized Q8_1 activations, outputting Q8_1
         *
         * This method takes Q8_1 blocks as input and produces Q8_1 blocks as output,
         * avoiding double quantization that would occur if using FP32 intermediate values.
         *
         * @param q8_1_activations Pre-quantized Q8_1 blocks [m, k/32]
         * @param C_q8_1 Output Q8_1 blocks buffer [m, (n+31)/32 blocks]
         * @param m Number of rows (sequence length)
         * @param n Output features (columns)
         * @param k Input features
         * @param bias Optional bias vector [n] to add before requantization (nullptr = no bias)
         * @param accumulate Unused (Q8_1 output doesn't support accumulation)
         * @param mpi_ctx MPI context for distributed execution
         * @param device_idx Device index for execution
         * @param workspace Optional pre-allocated workspace (nullptr = kernel allocates)
         *
         * @return true on success, false if not supported
         */
        virtual bool multiply_with_precomputed_q8_1_to_q8_1(
            const void *q8_1_activations,
            void *C_q8_1,
            int m, int n, int k,
            const TensorBase *bias = nullptr,
            bool accumulate = false,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            DeviceWorkspaceManager *workspace = nullptr)
        {
            // Default: not supported
            (void)q8_1_activations;
            (void)C_q8_1;
            (void)m;
            (void)n;
            (void)k;
            (void)bias;
            (void)accumulate;
            (void)mpi_ctx;
            (void)device_idx;
            (void)workspace;
            return false;
        }

        /**
         * @brief GEMM with pre-quantized Q8_1 input, outputting to Q16_1 format
         *
         * Performs C_q16 = A_q8 @ W, where A is pre-quantized Q8_1 and output is Q16_1.
         * This provides 256× more precision than Q8_1 output, critical for preserving
         * dynamic range in K projections for HybridQ16 attention.
         *
         * Use case: K projection in HybridQ16 mode where small values would be lost
         * in Q8_1 output due to high dynamic range (~130 max).
         *
         * @param q8_1_activations Pre-quantized Q8_1 activation blocks [m, k/32 blocks]
         * @param C_q16_1 Output Q16_1 blocks [m, n/block_size blocks] - must be pre-allocated
         * @param m Number of rows (sequence length)
         * @param n Output features (columns, must be divisible by block_size)
         * @param k Input features (rows in weight matrix)
         * @param q16_block_size Q16_1 block size: 32, 64, or 128 (should match head_dim)
         * @param bias Optional bias vector [n] to add before requantization (nullptr if none)
         * @param mpi_ctx MPI context
         * @param device_idx Device index
         * @param workspace Optional pre-allocated workspace (nullptr = kernel allocates)
         *
         * @return true on success
         */
        virtual bool multiply_with_precomputed_q8_1_to_q16_1(
            const void *q8_1_activations,
            void *C_q16_1,
            int m, int n, int k,
            int q16_block_size = 64,
            const TensorBase *bias = nullptr,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            DeviceWorkspaceManager *workspace = nullptr)
        {
            // Default: not supported
            (void)q8_1_activations;
            (void)C_q16_1;
            (void)m;
            (void)n;
            (void)k;
            (void)q16_block_size;
            (void)bias;
            (void)mpi_ctx;
            (void)device_idx;
            (void)workspace;
            return false;
        }

        /**
         * @brief Tensor-based activation-activation matmul: C = alpha * A @ B^T + beta * C
         *
         * Type-aware dispatch based on tensor native_type():
         * - FP32 × FP32 → FP32: Standard OpenBLAS/OneDNN path
         * - BF16 × BF16 → FP32: OneDNN bf16bf16f32
         * - FP16 × FP16 → FP32: OneDNN fp16fp16f32
         * - Q8_1 × Q8_1 → FP32: Dequant both, FP32 matmul (attention scores are always FP32)
         *
         * @param A Left activation tensor [m, k]
         * @param B Right activation tensor [n, k] if transpose_B, [k, n] otherwise
         * @param C Output tensor [m, n]
         * @param transpose_B Whether to transpose B (true for Q@K^T)
         * @param alpha Scale factor (e.g., 1/sqrt(d_k) for attention)
         * @param beta Scale factor for existing C
         * @param mpi_ctx MPI context (nullptr = single node)
         * @param device_idx Device index (-1 = CPU, ≥0 = GPU)
         *
         * @return true on success, false if type combination not supported
         *
         * @note Default implementation returns false.
         *       Subclasses that include Tensors.h can override with type-aware dispatch.
         */
        virtual bool multiply_activations_tensor(
            const TensorBase *A, const TensorBase *B, TensorBase *C,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            // Default: not supported (TensorBase is only forward-declared here)
            // Subclasses that include Tensors.h can override with real implementation
            (void)A;
            (void)B;
            (void)C;
            (void)transpose_B;
            (void)alpha;
            (void)beta;
            (void)mpi_ctx;
            (void)device_idx;
            return false;
        }

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
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)A;
            (void)B;
            (void)C;
            (void)m;
            (void)n;
            (void)k;
            (void)lda;
            (void)ldb;
            (void)ldc;
            (void)transpose_B;
            (void)alpha;
            (void)beta;
            (void)mpi_ctx;
            (void)device_idx;
            return false;
        }

        /**
         * @brief Virtual implementation for typed strided activation GEMM (type-erased)
         */
        virtual bool multiply_activations_strided_typed_impl(
            const void *A, const void *B, float *C,
            int m, int n, int k,
            int lda, int ldb, int ldc,
            bool transpose_B,
            float alpha, float beta,
            const IMPIContext *mpi_ctx,
            int device_idx,
            ActivationFormat format_A, ActivationFormat format_B)
        {
            (void)A;
            (void)B;
            (void)C;
            (void)m;
            (void)n;
            (void)k;
            (void)lda;
            (void)ldb;
            (void)ldc;
            (void)transpose_B;
            (void)alpha;
            (void)beta;
            (void)mpi_ctx;
            (void)device_idx;
            (void)format_A;
            (void)format_B;
            return false;
        }

        /**
         * @brief Template-based typed activation-activation GEMM
         *
         * Supports native-precision GEMM without forced FP32 conversion.
         * C = alpha * A @ B^T + beta * C (typed precision)
         *
         * **Precision Combinations**:
         * - (FP32, FP32) → FP32 GEMM (OpenBLAS/OneDNN)
         * - (BF16, BF16) → BF16 GEMM (OneDNN bf16bf16f32 on AVX512_BF16)
         * - (FP16, FP16) → FP16 GEMM (OneDNN fp16fp16f32)
         * - (INT8, INT8) → INT8 GEMM (OneDNN int8int8s32 on AVX512-VNNI)
         * - Mixed precisions → Convert to common format (lower precision preferred)
         *
         * @tparam ActT Activation element type (float, uint16_t, int8_t)
         * @tparam WeightT Weight element type (same options as ActT)
         *
         * @param A Left activation matrix [m, k]
         * @param B Right activation matrix [n, k] if transpose_B=true
         * @param C Output matrix [m, n] (always float* for now)
         * @param m Number of rows in A and C
         * @param n Number of rows in B (transpose_B=true) or cols (false)
         * @param k Number of columns in A and B (transpose_B=true)
         * @param transpose_B Whether to transpose B
         * @param alpha Scale factor for A@B
         * @param beta Scale factor for existing C
         * @param mpi_ctx MPI context (nullptr = single node)
         * @param device_idx Device index (-1 = CPU, ≥0 = GPU)
         * @param format Activation format hint (e.g. BF16 vs FP16 for uint16_t)
         *
         * @return true on success, false on error
         *
         * @note Default implementation returns false (not implemented).
         *       Kernels override to support specific precision combinations.
         */
        /**
         * @brief Template-based typed strided activation-activation GEMM
         *
         * Supports strided memory access with typed precision.
         * C = alpha * A @ B^T + beta * C (typed precision, strided)
         *
         * @tparam ActT Activation element type (float, uint16_t, int8_t)
         * @tparam WeightT Weight element type (same options as ActT)
         *
         * @param A Left activation matrix with stride lda
         * @param B Right activation matrix with stride ldb
         * @param C Output matrix with stride ldc (always float* for now)
         * @param m Number of rows in A and C
         * @param n Number of rows in B (transpose_B=true) or cols (false)
         * @param k Number of columns in A and B (transpose_B=true)
         * @param lda Leading dimension of A (stride between rows)
         * @param ldb Leading dimension of B (stride between rows)
         * @param ldc Leading dimension of C (stride between rows)
         * @param transpose_B Whether to transpose B
         * @param alpha Scale factor for A@B
         * @param beta Scale factor for existing C
         * @param mpi_ctx MPI context (nullptr = single node)
         * @param device_idx Device index (-1 = CPU, ≥0 = GPU)
         * @param format Activation format hint (e.g. BF16 vs FP16 for uint16_t)
         *
         * @return true on success, false on error
         *
         * @note Default implementation returns false (not implemented).
         *       Kernels override to support specific precision combinations.
         */
        template <typename ActT, typename WeightT>
        bool multiply_activations_strided_typed(
            const ActT *A, const WeightT *B, float *C,
            int m, int n, int k,
            int lda, int ldb, int ldc,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            ActivationFormat format = ActivationFormat::FP32)
        {
            ActivationFormat fmt_A = (std::is_same_v<ActT, float>) ? ActivationFormat::FP32 : format;
            ActivationFormat fmt_B = (std::is_same_v<WeightT, float>) ? ActivationFormat::FP32 : format;
            return multiply_activations_strided_typed_impl(
                static_cast<const void *>(A), static_cast<const void *>(B), C,
                m, n, k, lda, ldb, ldc, transpose_B, alpha, beta, mpi_ctx, device_idx, fmt_A, fmt_B);
        }

        /**
         * @brief Type-erased implementation for typed fused matmul+softmax with striding
         */
        virtual bool multiply_with_softmax_strided_typed_impl(
            const void *A, const void *B, float *C,
            int m, int n, int k,
            int lda, int ldb, int ldc,
            float scale,
            bool transpose_B,
            int softmax_axis,
            const float *mask,
            bool is_causal,
            const IMPIContext *mpi_ctx,
            int device_idx,
            ActivationFormat format_A,
            ActivationFormat format_B)
        {
            (void)A;
            (void)B;
            (void)C;
            (void)m;
            (void)n;
            (void)k;
            (void)lda;
            (void)ldb;
            (void)ldc;
            (void)scale;
            (void)transpose_B;
            (void)softmax_axis;
            (void)mask;
            (void)is_causal;
            (void)mpi_ctx;
            (void)device_idx;
            (void)format_A;
            (void)format_B;
            return false;
        }

        /**
         * @brief Template-based typed fused matmul+softmax with striding
         *
         * C = Softmax(A @ B^T / sqrt(k) + mask)
         *
         * @tparam ActT Activation element type (float, uint16_t, int8_t)
         * @tparam WeightT Weight element type (same options as ActT)
         *
         * @param A Left activation matrix [m, k]
         * @param B Right activation matrix [n, k] (transposed)
         * @param C Output matrix [m, n]
         * @param m Number of rows in A and C
         * @param n Number of rows in B
         * @param k Number of columns in A and B
         * @param transpose_B Whether to transpose B
         * @param softmax_axis Axis for softmax
         * @param mask Optional mask to add to logits before softmax
         * @param mpi_ctx MPI context
         * @param device_idx Device index
         * @param format Activation format hint (e.g. BF16 vs FP16 for uint16_t)
         *
         * @return true on success, false on error
         */
        template <typename ActT, typename WeightT>
        bool multiply_with_softmax_strided_typed(
            const ActT *A, const WeightT *B, float *C,
            int m, int n, int k,
            int lda, int ldb, int ldc,
            float scale = 1.0f,
            bool transpose_B = true,
            int softmax_axis = 1,
            const float *mask = nullptr,
            bool is_causal = false,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            ActivationFormat format = ActivationFormat::FP32)
        {
            ActivationFormat fmt_A = (std::is_same_v<ActT, float>) ? ActivationFormat::FP32 : format;
            ActivationFormat fmt_B = (std::is_same_v<WeightT, float>) ? ActivationFormat::FP32 : format;
            return multiply_with_softmax_strided_typed_impl(
                static_cast<const void *>(A), static_cast<const void *>(B), C,
                m, n, k, lda, ldb, ldc, scale, transpose_B, softmax_axis, mask, is_causal, mpi_ctx, device_idx, fmt_A, fmt_B);
        }

        // =============================================================================
        // Fused Multi-GEMM Interface (Activation Sharing)
        // =============================================================================
        // These methods enable fusing multiple GEMMs that share the same input activations.
        // For quantized kernels, this avoids redundant FP32→Q8_1 quantization passes.
        //
        // Use cases:
        // - FFN: gate + up projections share same input (FusedDualGEMM)
        // - Attention: Q + K + V projections share same input (FusedTripleGEMM)
        //
        // The workflow:
        // 1. Call quantize_activations() once to get reusable Q8_1 buffer
        // 2. Call multiply_with_precomputed_q8_1() N times, passing the same buffer
        // 3. Buffer can be a thread-local scratch to avoid allocation
        // =============================================================================

        /**
         * @brief Quantize FP32 activations to Q8_1 format for reuse across multiple GEMMs
         *
         * This method quantizes FP32 activations to Q8_1 blocks that can be reused
         * by multiple subsequent GEMM operations. This is the first step in the
         * fused multi-GEMM workflow.
         *
         * @param A Input FP32 activations [m, k]
         * @param q8_1_buffer Output buffer for Q8_1 blocks [m * k_blocks], where k_blocks = k/32
         *                    Must be pre-allocated with at least m * ((k+31)/32) * sizeof(Q8_1Block) bytes
         * @param m Number of rows in activation matrix
         * @param k Number of columns in activation matrix
         *
         * @return true on success, false if not supported (e.g., non-quantized kernel)
         *
         * @note This is a no-op for floating-point kernels that don't need quantization.
         *       For CPUQuantisedGemmKernel, this performs the FP32→Q8_1 conversion.
         *
         * @note Thread safety: This method is thread-safe. The output buffer can be
         *       thread-local to avoid contention.
         */
        virtual bool quantize_activations(
            const float *A,
            void *q8_1_buffer,
            int m, int k)
        {
            (void)A;
            (void)q8_1_buffer;
            (void)m;
            (void)k;
            return false; // Not implemented for non-quantized kernels
        }

        /**
         * @brief Get required buffer size for quantized activations
         *
         * @param m Number of rows in activation matrix
         * @param k Number of columns in activation matrix
         * @return Required buffer size in bytes, or 0 if quantization not supported
         */
        virtual size_t get_quantized_activation_buffer_size(int m, int k) const
        {
            (void)m;
            (void)k;
            return 0; // Not supported for non-quantized kernels
        }

        /**
         * @brief GEMM with pre-quantized activations (no redundant quantization)
         *
         * This method performs GEMM using pre-quantized Q8_1 activations from a prior
         * quantize_activations() call. This is the second step in the fused multi-GEMM
         * workflow and enables activation sharing across multiple GEMMs.
         *
         * Performance: When calling N GEMMs with the same input, this saves (N-1)
         * quantization passes compared to calling multiply_tensor() N times.
         *
         * @param q8_1_activations Pre-quantized Q8_1 blocks from quantize_activations()
         * @param C Output matrix [m, n] (FP32)
         * @param m Number of rows in activation matrix
         * @param n Number of columns in output (rows in weight matrix)
         * @param k Number of columns in activation matrix
         * @param bias Optional bias vector [n] to add after GEMM (nullptr = no bias)
         * @param accumulate If true, add to existing C; if false, overwrite
         * @param alpha Scale factor for GEMM result
         * @param beta Scale factor for existing C (only used if accumulate=true)
         * @param mpi_ctx MPI context for distributed execution
         * @param device_idx Device index (-1 = CPU)
         * @param workspace Optional pre-allocated workspace (nullptr = kernel allocates)
         *
         * @return true on success, false if not supported
         *
         * @note For non-quantized kernels, this returns false. Use multiply_tensor() instead.
         * @note The q8_1_activations buffer must have been populated by quantize_activations()
         *       with the same m and k dimensions.
         */
        virtual bool multiply_with_precomputed_q8_1(
            const void *q8_1_activations,
            float *C,
            int m, int n, int k,
            const TensorBase *bias = nullptr,
            bool accumulate = false,
            float alpha = 1.0f, float beta = 0.0f,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            DeviceWorkspaceManager *workspace = nullptr)
        {
            // Delegate to fused version with no fusion
            return multiply_with_precomputed_q8_1(
                q8_1_activations, C, m, n, k, bias, accumulate, alpha, beta,
                mpi_ctx, device_idx, GemmFusedOps::none(), workspace);
        }

        /**
         * @brief GEMM with pre-quantized activations and fused post-ops
         *
         * Extended version that supports fused operations (SwiGLU, Softmax, etc.)
         * after the GEMM computation. This enables single-pass execution of
         * common patterns like FFN (GEMM + SwiGLU) and attention (GEMM + Softmax).
         *
         * @param q8_1_activations Pre-quantized Q8_1 blocks from quantize_activations()
         * @param C Output matrix [m, n] (FP32)
         * @param m Number of rows in activation matrix
         * @param n Number of columns in output (rows in weight matrix)
         * @param k Number of columns in activation matrix
         * @param bias Optional bias vector [n] to add after GEMM (nullptr = no bias)
         * @param accumulate If true, add to existing C; if false, overwrite
         * @param alpha Scale factor for GEMM result
         * @param beta Scale factor for existing C (only used if accumulate=true)
         * @param mpi_ctx MPI context for distributed execution
         * @param device_idx Device index (-1 = CPU)
         * @param fused_ops Fused operation configuration (SwiGLU, Softmax, etc.)
         * @param workspace Optional pre-allocated workspace (nullptr = kernel allocates)
         *
         * @return true on success, false if not supported
         *
         * Example - SwiGLU fusion for FFN:
         * @code
         * // Compute gate projection first
         * gate_kernel->multiply_with_precomputed_q8_1(q8_acts, gate_out, m, n, k);
         *
         * // Compute up projection with fused SwiGLU
         * up_kernel->multiply_with_precomputed_q8_1(q8_acts, up_out, m, n, k,
         *     nullptr, false, 1.0f, 0.0f, nullptr, -1,
         *     GemmFusedOps::swiglu(gate_out));
         * @endcode
         */
        virtual bool multiply_with_precomputed_q8_1(
            const void *q8_1_activations,
            float *C,
            int m, int n, int k,
            const TensorBase *bias,
            bool accumulate,
            float alpha, float beta,
            const IMPIContext *mpi_ctx,
            int device_idx,
            const GemmFusedOps &fused_ops,
            DeviceWorkspaceManager *workspace = nullptr)
        {
            (void)q8_1_activations;
            (void)C;
            (void)m;
            (void)n;
            (void)k;
            (void)bias;
            (void)accumulate;
            (void)alpha;
            (void)beta;
            (void)mpi_ctx;
            (void)device_idx;
            (void)fused_ops;
            (void)workspace;
            return false; // Not implemented for non-quantized kernels
        }

        /**
         * @brief Check if this kernel supports the fused multi-GEMM interface
         *
         * @return true if quantize_activations() and multiply_with_precomputed_q8_1()
         *         are implemented
         */
        virtual bool supports_activation_sharing() const
        {
            return false; // Default: not supported
        }

        // =============================================================================
        // Weight Lifecycle Management
        //
        // Enables decoupling packed weights from GEMM engines for:
        // - Cross-NUMA expert migration (clone packed data to destination socket)
        // - Memory reclamation (release departed expert weights)
        // - Cross-device transfer (CPU ↔ GPU, phase 2)
        //
        // Default implementations are no-ops for backward compatibility.
        // Override in kernels that own packed weight data.
        // =============================================================================

        /**
         * @brief Detach packed weights from this kernel, transferring ownership.
         *
         * The kernel becomes invalid for computation after this call (hasWeights()
         * returns false). The caller receives a transferable IPackedWeights that
         * can be cloned to another NUMA node or converted to another device format.
         *
         * @return Owned packed weights, or nullptr if not supported / no weights.
         */
        virtual std::unique_ptr<IPackedWeights> detachWeights()
        {
            return nullptr; // Default: not supported
        }

        /**
         * @brief Clone packed weights without invalidating this kernel.
         *
         * Used for non-destructive expert replication/migration. Implementations
         * should include any source-format side payload needed to convert to a
         * different prepared format later (for example native quantized blocks
         * alongside CPU VNNI metadata).
         */
        virtual std::unique_ptr<IPackedWeights> cloneWeights() const
        {
            return nullptr; // Default: not supported
        }

        /**
         * @brief Attach pre-packed weights to this kernel.
         *
         * Replaces any existing weights. The kernel takes ownership of the data.
         * After a successful attach, hasWeights() returns true and the kernel
         * is ready for computation.
         *
         * @param weights Pre-packed weights (must be compatible format)
         * @return true if attached successfully, false if format mismatch
         */
        virtual bool attachWeights(std::unique_ptr<IPackedWeights> weights)
        {
            (void)weights;
            return false; // Default: not supported
        }

        /**
         * @brief Release all packed weight data, freeing memory.
         *
         * The kernel becomes invalid for computation after this call.
         * Unlike detachWeights(), the data is destroyed rather than returned.
         * Use for departed experts whose weights are no longer needed.
         */
        virtual void releaseWeights() {}

        /**
         * @brief Check if this kernel has valid packed weights for computation.
         *
         * Returns false after detachWeights() or releaseWeights() until
         * new weights are attached via attachWeights().
         *
         * @return true if kernel has valid weights and can compute
         */
        virtual bool hasWeights() const { return true; }

        /**
         * @brief Bytes of owned packed/prepared weight data held by this engine.
         *
         * This is a non-mutating accounting hook. Implementations should not
         * clone, detach, or materialize lazy state just to answer this query.
         */
        virtual size_t packedWeightBytes() const { return 0; }

        // =============================================================================
        // Weight Dimension Accessors (for Tensor Parallelism)
        // =============================================================================
        // These methods expose the actual dimensions of the packed weight matrix.
        // Essential for tensor parallelism where weights may be sharded:
        // - Column-parallel: N dimension is split across ranks
        // - Row-parallel: K dimension is split across ranks
        //
        // Callers can query these to determine actual local dimensions when weights
        // have been sharded, rather than assuming the full model dimensions.
        // =============================================================================

        /**
         * @brief Get the output dimension (N) of the weight matrix
         *
         * For sharded column-parallel weights (QKV, Gate/Up), this returns the
         * local N dimension (n_full / world_size), not the full model dimension.
         *
         * @return Number of output features this kernel produces, or 0 if unknown
         */
        virtual int get_n() const { return 0; }

        /**
         * @brief Get the input dimension (K) of the weight matrix
         *
         * For sharded row-parallel weights (Wo, Down), this returns the
         * local K dimension (k_full / world_size), not the full model dimension.
         *
         * @return Number of input features this kernel expects, or 0 if unknown
         */
        virtual int get_k() const { return 0; }
    };

    // Note: TensorBase alias already declared at top of file

    // ==========================================================================
    // Attention Mode Detection
    // ==========================================================================

    /**
     * @brief Attention computation mode for kernel dispatch
     *
     * Different attention scenarios require different kernel implementations
     * for optimal performance:
     *
     * - **PREFILL**: Processing a full prompt (GEMM-based, high parallelism)
     * - **DECODE**: Single new token attending to cache (dot product + softmax)
     * - **BATCHED_DECODE**: Multiple sequences, each decoding one token
     * - **CHUNKED_PREFILL**: Large prompt split into chunks (for memory limits)
     *
     * Kernels use this enum for internal dispatch to specialized implementations.
     */
    enum class AttentionMode
    {
        PREFILL,        ///< seq_len > 1, kv_len == seq_len (batch GEMM path)
        DECODE,         ///< seq_len == 1, single sequence (dot product path)
        BATCHED_DECODE, ///< batch_size > 1, seq_len == 1 (parallel decode)
        CHUNKED_PREFILL ///< seq_len > 1, kv_len > seq_len (incremental prefill)
    };

    /**
     * @brief Detect attention mode from dimensions
     *
     * Used by kernels to dispatch to optimized implementations.
     *
     * @param batch_size Number of sequences
     * @param seq_len Query sequence length (new tokens)
     * @param kv_len Key/Value length (total cached + new tokens)
     * @return Detected AttentionMode
     *
     * Logic:
     * - batch_size > 1 && seq_len == 1 → BATCHED_DECODE
     * - seq_len == 1 → DECODE
     * - seq_len < kv_len → CHUNKED_PREFILL
     * - otherwise → PREFILL
     */
    inline AttentionMode detect_attention_mode(int batch_size, int seq_len, int kv_len)
    {
        if (batch_size > 1 && seq_len == 1)
            return AttentionMode::BATCHED_DECODE;
        if (seq_len == 1)
            return AttentionMode::DECODE;
        if (seq_len < kv_len)
            return AttentionMode::CHUNKED_PREFILL;
        return AttentionMode::PREFILL;
    }

    /**
     * @brief Get string name for AttentionMode (for logging)
     */
    inline const char *attention_mode_name(AttentionMode mode)
    {
        switch (mode)
        {
        case AttentionMode::PREFILL:
            return "PREFILL";
        case AttentionMode::DECODE:
            return "DECODE";
        case AttentionMode::BATCHED_DECODE:
            return "BATCHED_DECODE";
        case AttentionMode::CHUNKED_PREFILL:
            return "CHUNKED_PREFILL";
        default:
            return "UNKNOWN";
        }
    }

    /**
     * @brief Attention computation kernel interface
     *
     * Computes scaled dot-product attention with optional causal masking:
     *   output = softmax(Q @ K^T / sqrt(d_k) + mask) @ V
     *
     * Supports:
     * - Grouped Query Attention (GQA): n_heads != n_kv_heads
     * - Multi-Query Attention (MQA): n_kv_heads == 1
     * - Causal masking with optional sliding window
     * - Multiple activation precisions via typed implementations
     *
     * Implementations:
     * - CPUAttentionKernelT<Precision>: Precision-enum based kernel
     */
    class ITensorAttention : public ITensorKernel
    {
    public:
        /**
         * @brief Compute attention using tensor objects with automatic type dispatch
         *
         * Inspects Q/K/V/output tensor native_type() and dispatches to the appropriate
         * typed method (compute, compute_batch, compute_q8_1, compute_batch_q8_1).
         *
         * @param Q Query tensor [batch_size * seq_len, n_heads * head_dim]
         * @param K Key tensor [batch_size * kv_len, n_kv_heads * head_dim]
         * @param V Value tensor [batch_size * kv_len, n_kv_heads * head_dim]
         * @param output Output tensor [batch_size * seq_len, n_heads * head_dim]
         * @param batch_size Batch size (1 for single sequence)
         * @param seq_len Query sequence length
         * @param kv_len Key/value sequence length (may differ from seq_len in decode)
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads (GQA: n_heads % n_kv_heads == 0)
         * @param head_dim Dimension per head
         * @param causal Apply causal (lower-triangular) masking
         * @param window_size Sliding window size (-1 = disabled)
         * @param workspace_scores Workspace for attention scores
         * @param workspace_mask Pre-built attention mask or nullptr
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index
         * @param head_start First query head to compute (0-indexed, default 0)
         * @param local_n_heads Number of query heads to compute (-1 = all)
         * @param local_n_kv_heads Number of KV heads for this slice (-1 = all)
         * @return true on success, false on failure or unsupported type combination
         *
         * @note Default returns false. Subclasses should override with type-aware dispatch.
         *
         * Tensor Parallelism Usage:
         * When distributing attention across ranks, each rank computes a subset of heads:
         *   Rank 0: head_start=0, local_n_heads=n_heads/2
         *   Rank 1: head_start=n_heads/2, local_n_heads=n_heads/2
         *
         * The output tensor is indexed as output[seq, head, dim], so each rank writes
         * to output[:, head_start:head_start+local_n_heads, :]. After computation,
         * an AllGather is needed to combine results (handled by caller).
         *
         * Note: Uses ITensor* (not TensorBase*) to support both CPU tensors and GPU
         * tensor wrappers (e.g., GpuTensorView from KV cache). Implementations extract
         * GPU pointers via gpu_data_ptr() or CPU data via dynamic_cast to TensorBase.
         *
         * @note This is the PRIMARY attention interface. All implementations must provide this.
         */
        virtual bool compute_tensor(
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
            int head_start = 0,        ///< First query head (TP slice start)
            int local_n_heads = -1,    ///< Number of query heads (-1 = all)
            int local_n_kv_heads = -1, ///< Number of KV heads (-1 = all)
            int gqa_n_rep = 0)         ///< Global GQA repetition factor (0 = auto from n_heads/n_kv_heads)
            = 0;

        /**
         * @brief Compute MTP verifier rows through a grouped decode-equivalent path.
         *
         * The verifier input contains M compact rows (currently production M=2..4)
         * appended after an existing prefix in the KV cache. The implementation
         * must compute every row as if serial decode had processed rows
         * `[0, M)` one at a time, including causal visibility:
         *
         * - row 0 may attend through `base_kv_len + 1`
         * - row 1 may attend through `base_kv_len + 2`
         * - ...
         *
         * This is a performance contract, not merely a correctness contract.
         * Implementations must not hide M ordinary one-token `compute_tensor()`
         * calls behind this method. Unsupported backends should return false so
         * stages can fail closed instead of silently taking a serial fallback.
         */
        virtual bool compute_verifier_rows_decode_equivalent(
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
            int gqa_n_rep = 0)
        {
            (void)Q;
            (void)K;
            (void)V;
            (void)output;
            (void)verifier_rows;
            (void)kv_len;
            (void)n_heads;
            (void)n_kv_heads;
            (void)head_dim;
            (void)causal;
            (void)window_size;
            (void)mpi_ctx;
            (void)device_idx;
            (void)head_start;
            (void)gqa_n_rep;
            return false;
        }

        /**
         * @brief Update attention device params stored in pinned host memory for graph replay
         *
         * Graph-captured attention reads AttentionDeviceParams from device memory.
         * Implementations update that device buffer before beginCapture()/graph
         * replay on the explicit stage stream; captured stage bodies must not
         * record H2D nodes.
         *
         * @param kv_len Number of cached tokens (including tokens to be appended this step)
         * @param position_offset Position offset for the current decode step
         */
        virtual void setDynamicAttnParams(int kv_len, int position_offset)
        {
            (void)kv_len;
            (void)position_offset;
        }

        virtual void setDynamicAttnParams(int kv_len, int position_offset, int query_rows)
        {
            (void)query_rows;
            setDynamicAttnParams(kv_len, position_offset);
        }

        /**
         * @brief Prepare device-side attention params before graph capture/replay.
         *
         * Backends with graph-captured attention should update any device-resident
         * scalar params on the explicit execution stream before beginCapture() or
         * graph launch. The default preserves existing backends by updating their
         * host-side params and reporting success.
         */
        virtual bool prepareDynamicAttnParams(
            int kv_len, int position_offset, int query_rows, void *stream)
        {
            (void)stream;
            setDynamicAttnParams(kv_len, position_offset, query_rows);
            return true;
        }

        /**
         * @brief Prepare attention params from device-owned sequence metadata.
         *
         * This graph-capturable path is used when the KV cache owns the live
         * token count on device. Implementations should enqueue a small kernel
         * on @p stream that derives AttentionDeviceParams from
         * @p post_append_cached_tokens_device after the KV append stage has
         * advanced the cache count. The default returns false so unsupported
         * backends fail hard instead of silently falling back to host-owned
         * scalar uploads.
         *
         * @param post_append_cached_tokens_device Device pointer to the current
         *        cached-token count for this layer/sequence after append.
         * @param seq_len Logical query row count for this attention stage.
         * @param query_rows Number of row-local dynamic attention params needed.
         * @param stream Explicit non-null backend stream.
         */
        virtual bool prepareDynamicAttnParamsFromDeviceSequenceState(
            const int *post_append_cached_tokens_device,
            int seq_len,
            int query_rows,
            void *stream)
        {
            (void)post_append_cached_tokens_device;
            (void)seq_len;
            (void)query_rows;
            (void)stream;
            return false;
        }
    };

    // =========================================================================
    // Forward Declarations for Fused Attention
    // =========================================================================

    namespace gemm
    {
        struct QuantisedPackedWeights;
    }

    // =========================================================================
    // ITensorFusedAttentionWo - Fused Attention + Wo Projection Interface
    // =========================================================================

    /**
     * @brief Parameters for fused attention + Wo projection + residual add
     *
     * This struct bundles all parameters for a fully-fused attention block that
     * combines:
     * 1. Q×K^T → attention scores (INT32 or FP32 domain)
     * 2. Softmax → attention weights
     * 3. Weights×V → attention context
     * 4. Context×Wo → output projection
     * 5. Output + residual → final output
     *
     * Design Goals:
     * - **Register-resident intermediate values**: No round-trip to memory between steps
     * - **Integer-domain pipeline**: Q16_1 implementation keeps everything in INT32
     * - **Flexible I/O**: Supports various quantized formats (Q16_1, Q8_1, FP32)
     *
     * Execution paths (determined by implementation):
     * - **Flash Decode** (seq_len_q=1): Single query, full KV, no tiling
     * - **FA2 Prefill** (seq_len_q>1): Batched queries, tiled KV processing
     *
     * @note Q, K, V use Q16BlockPtr for type-safe access to quantized blocks.
     *       Residual pointers are typed as Q16_1Block* (always 32-element blocks).
     */
    struct FusedAttentionWoParams
    {
        // ============== Attention tensors (Q16_1 blocks, type-safe) ==============

        Q16BlockPtr Q; ///< Query tensor [seq_len_q × n_heads × head_dim]
        Q16BlockPtr K; ///< Key tensor [kv_len × n_kv_heads × head_dim]
        Q16BlockPtr V; ///< Value tensor [kv_len × n_kv_heads × head_dim]

        // ============== Wo projection weights ==============

        /**
         * @brief VNNI-packed Wo weights for VPDPWSSD (INT16×INT16→INT32)
         *
         * Created from Q8_1 via KernelFactory::ensurePackedWeightsInTensorCache()
         * For FP32 implementations, this may be nullptr (use FP32 Wo instead)
         */
        const gemm::QuantisedPackedWeights *Wo_packed = nullptr;

        /**
         * @brief FP32 Wo weights [d_model × d_model] for FP32 implementations
         *
         * Used when Wo_packed is nullptr (pure FP32 path)
         */
        const float *Wo_fp32 = nullptr;

        // ============== Residual tensors (32-element Q16_1Block, typed) ==============

        const Q16_1Block *residual_in = nullptr; ///< Input residual [seq_len_q × d_model]
        Q16_1Block *residual_out = nullptr;      ///< Output residual [seq_len_q × d_model]
                                                 ///< Can be same as residual_in for in-place

        // ============== Dimensions ==============

        int seq_len_q = 0;  ///< Number of query positions
        int kv_len = 0;     ///< Number of KV positions (cache length)
        int n_heads = 0;    ///< Number of query heads
        int n_kv_heads = 0; ///< Number of KV heads (for GQA/MQA)
        int head_dim = 0;   ///< Dimension per head (typically 64 or 128)
        int d_model = 0;    ///< Model dimension (n_heads * head_dim)

        // ============== KV Cache Layout ==============

        /**
         * @brief Stride in positions between consecutive KV heads
         *
         * For HEAD_MAJOR sparse cache: max_seq_len (sparse allocation)
         * For dense/transposed data: kv_len (packed)
         * If 0, defaults to kv_len (dense layout assumption)
         */
        int kv_head_stride = 0;

        // ============== Attention configuration ==============

        float scale = 0.0f;      ///< 1/sqrt(head_dim), auto-computed if 0
        bool causal = true;      ///< Apply causal masking
        int position_offset = 0; ///< Position offset for causal mask (decode)
        int window_size = -1;    ///< Sliding window size (-1 = disabled)

        // ============== Tiling parameters (for FA2 prefill) ==============

        int Bc = 256; ///< KV tile size for FA2 prefill (ignored for decode)
        int Br = 16;  ///< Query tile size for FA2 prefill

        // ============== Snapshot/Debug buffers ==============

        /**
         * @brief Optional FP32 buffer for attention scores snapshot
         *
         * If non-null, attention scores (Q×K^T × scale) will be dequantized
         * to FP32 and stored here [seq_len_q × n_heads × kv_len]
         * for debugging/parity testing.
         */
        float *scores_snapshot = nullptr;

        /**
         * @brief Optional FP32 buffer for attention context snapshot
         *
         * If non-null, context (softmax × V) will be dequantized to FP32
         * and stored here [seq_len_q × n_heads × head_dim]
         * before Wo projection.
         */
        float *context_snapshot = nullptr;

        /**
         * @brief Optional FP32 buffer for Wo projection output snapshot
         *
         * If non-null, Wo output will be dequantized to FP32
         * and stored here [seq_len_q × d_model]
         * before residual add.
         */
        float *wo_output_snapshot = nullptr;

        /**
         * @brief Optional FP32 buffer for attention residual output snapshot
         *
         * If non-null, the final output (after residual add) will be
         * dequantized to FP32 and stored here [seq_len_q × d_model].
         * Corresponds to ATTENTION_RESIDUAL in the FP32 pipeline.
         */
        float *attention_residual_snapshot = nullptr;

        // ============== HybridQ16 K Precision Fix: Per-head scales ==============

        /**
         * @brief Per-head dynamic scales for Q vectors (HybridQ16 K precision fix)
         *
         * For HybridQ16 mode where Q uses fixed kv_cache_scale, this is nullptr.
         * For future dynamic-scale Q support, this would be [seq_len_q * n_heads].
         * Shape: [seq_len_q * n_heads] when non-null
         */
        const float *q_head_scales = nullptr;

        /**
         * @brief Per-head dynamic scales for K vectors (HybridQ16 K precision fix)
         *
         * When GEMM outputs K as Q16_1 (instead of Q8_1) and RoPE uses dynamic
         * per-head scale, these scales must be provided to the attention kernel
         * for correct Q×K^T computation.
         *
         * Shape: [kv_len * n_kv_heads] when non-null
         * - For prefill: [seq_len * n_kv_heads] with fresh K vectors
         * - For decode: scales are stored in KV cache alongside K data
         *
         * When nullptr, K is assumed to have fixed kv_cache_scale (standard HybridQ16).
         */
        const float *k_head_scales = nullptr;

        // ============== Optional debug metadata ==============

        /**
         * @brief Layer index for debug/snapshot correlation
         *
         * Not required for kernel correctness. When set, debug instrumentation can
         * gate dumps to a specific layer to keep logs manageable.
         */
        int layer_idx = -1;

        // ============== Helper methods ==============

        /**
         * @brief Get the attention scale (auto-compute if not set)
         */
        float get_scale() const
        {
            if (scale > 0.0f)
                return scale;
            return 1.0f / std::sqrt(static_cast<float>(head_dim));
        }

        /**
         * @brief Check if this is a decode pass (single query token)
         */
        bool is_decode() const { return seq_len_q == 1; }

        /**
         * @brief Get the KV head index for a given query head (GQA mapping)
         */
        int get_kv_head(int query_head) const
        {
            if (n_kv_heads == n_heads)
                return query_head;
            return query_head / (n_heads / n_kv_heads);
        }

        /**
         * @brief Get the block size from Q tensor (Q/K/V must use same block size)
         */
        Q16BlockSize block_size() const { return Q.block_size; }

        /**
         * @brief Validate that Q, K, V all have the same block size
         */
        bool validate_block_sizes() const
        {
            if (Q.empty() || K.empty() || V.empty())
                return false;
            return Q.block_size == K.block_size && K.block_size == V.block_size;
        }

        /**
         * @brief Get blocks per row for attention computation
         */
        int blocks_per_row() const
        {
            int bs = static_cast<int>(Q.block_size);
            return (head_dim + bs - 1) / bs;
        }
    };

    /**
     * @brief Interface for fused attention + Wo projection + residual kernels
     *
     * This interface is designed for high-performance fused attention kernels
     * that combine the entire attention block (QKV → softmax → context → Wo → residual)
     * into a single kernel call, maximizing register utilization.
     *
     * Primary use case: **Integer-domain Q16_1 attention**
     * - Q×K^T → INT32 scores
     * - Exp2FixedSoftmax → INT16 weights
     * - Weights×V → INT32 context accumulators
     * - Context×Wo (VPDPWSSD) → INT32 → Q16_1
     * - Q16_1 + Q16_1 residual → Q16_1 output
     *
     * The interface also supports FP32 implementations for reference/fallback.
     *
     * **Why separate from ITensorAttention?**
     * ITensorAttention outputs attention context only. This interface outputs
     * the full attention block result (after Wo and residual), enabling:
     * - Register-resident Wo projection (no context materialize to memory)
     * - Native quantized residual add (no FP32 conversion)
     * - Single kernel launch for entire attention block
     *
     * Implementations:
     * - Q16FusedAttentionRef: Scalar C++ reference (for testing/debugging)
     * - Q16FusedAttentionJit: AVX512 JIT (production)
     * - FP32FusedAttentionRef: FP32 fallback (parity testing)
     */
    class ITensorFusedAttentionWo : public ITensorKernel
    {
    public:
        /**
         * @brief Execute fused attention + Wo + residual
         *
         * Dispatches to decode or prefill path based on params.seq_len_q.
         *
         * @param params Complete parameter struct (see FusedAttentionWoParams)
         * @param mpi_ctx MPI context for distributed execution (optional)
         * @param device_idx Device index (-1 = CPU)
         * @return true on success, false on validation/execution error
         *
         * @note This is the primary entry point. Implementations may override
         *       compute_decode() and compute_prefill() for path-specific optimizations.
         */
        virtual bool compute(
            const FusedAttentionWoParams &params,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;

        /**
         * @brief Execute fused attention for decode (single query token)
         *
         * Optimized for autoregressive token generation:
         * - Single query against full KV cache
         * - Parallel over KV positions, no tiling
         * - Latency-optimized
         *
         * @param params Parameters (must have seq_len_q == 1)
         * @param mpi_ctx MPI context
         * @param device_idx Device index
         * @return true on success
         *
         * Default: calls compute(params, mpi_ctx, device_idx) with validation
         */
        virtual bool compute_decode(
            const FusedAttentionWoParams &params,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            if (params.seq_len_q != 1)
                return false;
            return compute(params, mpi_ctx, device_idx);
        }

        /**
         * @brief Execute fused attention for prefill (multiple query tokens)
         *
         * Optimized for prompt processing:
         * - Batched queries with tiled KV processing
         * - Throughput-optimized (FA2 algorithm)
         *
         * @param params Parameters (typically seq_len_q > 1)
         * @param mpi_ctx MPI context
         * @param device_idx Device index
         * @return true on success
         *
         * Default: calls compute(params, mpi_ctx, device_idx) with validation
         */
        virtual bool compute_prefill(
            const FusedAttentionWoParams &params,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            if (params.seq_len_q < 1)
                return false;
            return compute(params, mpi_ctx, device_idx);
        }

        /**
         * @brief Execute using TensorBase objects with automatic type dispatch
         *
         * Inspects tensor native_type() and constructs appropriate params.
         * This is the type-safe interface for stages that don't know the
         * concrete tensor format.
         *
         * @param Q Query tensor [seq_len_q × n_heads × head_dim]
         * @param K Key tensor [kv_len × n_kv_heads × head_dim]
         * @param V Value tensor [kv_len × n_kv_heads × head_dim]
         * @param Wo_tensor Wo weight tensor (for extracting packed weights or FP32 data)
         * @param residual_in Input residual tensor [seq_len_q × d_model]
         * @param residual_out Output residual tensor [seq_len_q × d_model]
         * @param seq_len_q Query sequence length
         * @param kv_len KV sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of KV heads
         * @param head_dim Head dimension
         * @param causal Apply causal masking
         * @param position_offset Position offset for decode
         * @param scores_snapshot Optional FP32 buffer for scores
         * @param context_snapshot Optional FP32 buffer for context
         * @param wo_output_snapshot Optional FP32 buffer for Wo output
         * @param mpi_ctx MPI context
         * @param device_idx Device index
         * @return true on success, false if tensor types not supported
         */
        virtual bool compute_tensor(
            const TensorBase *Q,
            const TensorBase *K,
            const TensorBase *V,
            const TensorBase *Wo_tensor,
            const TensorBase *residual_in,
            TensorBase *residual_out,
            int seq_len_q,
            int kv_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            bool causal = true,
            int position_offset = 0,
            float *scores_snapshot = nullptr,
            float *context_snapshot = nullptr,
            float *wo_output_snapshot = nullptr,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            // Default: not supported (subclasses override with type-aware dispatch)
            (void)Q;
            (void)K;
            (void)V;
            (void)Wo_tensor;
            (void)residual_in;
            (void)residual_out;
            (void)seq_len_q;
            (void)kv_len;
            (void)n_heads;
            (void)n_kv_heads;
            (void)head_dim;
            (void)causal;
            (void)position_offset;
            (void)scores_snapshot;
            (void)context_snapshot;
            (void)wo_output_snapshot;
            (void)mpi_ctx;
            (void)device_idx;
            return false;
        }

        /**
         * @brief Get input precision format supported by this kernel
         *
         * Returns the activation precision this kernel expects for Q/K/V inputs.
         * Used by stages to verify tensor format compatibility.
         */
        virtual ActivationFormat input_format() const = 0;

        /**
         * @brief Get output precision format produced by this kernel
         *
         * Returns the activation precision of the residual output.
         */
        virtual ActivationFormat output_format() const = 0;

        /**
         * @brief Check if this kernel requires VNNI-packed Wo weights
         *
         * @return true if Wo_packed must be provided, false if Wo_fp32 is used
         */
        virtual bool requires_packed_wo() const = 0;
    };

    /**
     * @brief Interface for fused Gate/Up projection GEMM (FFN)
     *
     * Performs two concurrent GEMM operations for SwiGLU FFN:
     * gate = input × W_gate [m, n_intermediate]
     * up   = input × W_up   [m, n_intermediate]
     *
     * These outputs are then used by SwiGLU: output = gate * silu(up)
     *
     * This interface wraps two ITensorGemm kernels internally, providing
     * a unified interface for the FusedGateUpGEMMStage.
     *
     * Workspace Support:
     * This interface also inherits IWorkspaceConsumer to forward workspace
     * binding to the underlying GEMM kernels (required for ROCm kernels).
     */
    class ITensorFusedGateUpGemm : public ITensorKernel, public IWorkspaceConsumer
    {
    public:
        /**
         * @brief Execute fused Gate/Up GEMM with tensor inputs/outputs
         *
         * @param input Input activations tensor [m, k]
         * @param output_gate Gate output buffer [m, n_gate]
         * @param output_up Up output buffer [m, n_up]
         * @param m Batch size (rows)
         * @param k Input dimension
         * @param n_gate Gate output dimension (n_intermediate)
         * @param n_up Up output dimension (should equal n_gate)
         * @param ctx Device context (optional)
         * @param device_idx Device index for kernel execution
         * @return true on success
         */
        virtual bool execute(
            const TensorBase *input,
            TensorBase *output_gate,
            TensorBase *output_up,
            int m, int k, int n_gate, int n_up,
            IDeviceContext *ctx = nullptr,
            int device_idx = -1) = 0;

        /**
         * @brief Execute fused Gate/Up GEMM with bias tensors
         *
         * @param input Input activations tensor [m, k]
         * @param output_gate Gate output buffer [m, n_gate]
         * @param output_up Up output buffer [m, n_up]
         * @param bias_gate Optional bias tensor for gate projection (nullptr if none)
         * @param bias_up Optional bias tensor for up projection (nullptr if none)
         * @param m Batch size (rows)
         * @param k Input dimension
         * @param n_gate Gate output dimension
         * @param n_up Up output dimension
         * @param ctx Device context (optional)
         * @param device_idx Device index
         * @return true on success
         */
        virtual bool execute_with_bias(
            const TensorBase *input,
            TensorBase *output_gate,
            TensorBase *output_up,
            const TensorBase *bias_gate,
            const TensorBase *bias_up,
            int m, int k, int n_gate, int n_up,
            IDeviceContext *ctx = nullptr,
            int device_idx = -1) = 0;
    };

    /**
     * @brief Interface for fused QKV projection GEMM
     *
     * Performs three concurrent GEMM operations for Q, K, V projections:
     * Q = input × Wq [m, n_q]
     * K = input × Wk [m, n_k]
     * V = input × Wv [m, n_v]
     *
     * Supports both FP32 and Q8_1 input activations, with mixed-precision
     * output (Q8_1 for Q/V, Q16_1 for K in HybridQ16 mode).
     *
     * Key benefits:
     * - Single activation quantization shared across all three GEMMs
     * - Better cache locality (input stays hot across projections)
     * - Native support for mixed-precision K output (Q16_1 for precision)
     */
    class ITensorFusedQKVGemm : public ITensorKernel
    {
    public:
        /**
         * @brief Execute fused QKV GEMM with FP32 input
         *
         * @param input_fp32 Input activations [m, k]
         * @param output_q Q output buffer (Q8_1Block* or float*)
         * @param output_k K output buffer (Q16_1Block* for mixed mode, or Q8_1Block*)
         * @param output_v V output buffer (Q8_1Block* or float*)
         * @param bias_q Q bias tensor (optional, nullptr if none)
         * @param bias_k K bias tensor (optional)
         * @param bias_v V bias tensor (optional)
         * @param m Batch size (rows)
         * @param n_q Q output columns
         * @param n_k K output columns
         * @param k Input/weight inner dimension
         * @param k_block_size Block size for K quantization (32, 64, or 128)
         * @return true on success
         */
        virtual bool execute_fp32(
            const float *input_fp32,
            void *output_q,
            void *output_k,
            void *output_v,
            const TensorBase *bias_q,
            const TensorBase *bias_k,
            const TensorBase *bias_v,
            int m, int n_q, int n_k, int k,
            int k_block_size = 64) = 0;

        /**
         * @brief Execute fused QKV GEMM with Q8_1 input
         *
         * @param input_q8_1 Pre-quantized input [m, k] as Q8_1 blocks
         * @param output_q Q output buffer (Q8_1Block*)
         * @param output_k K output buffer (Q16_1Block* for mixed mode)
         * @param output_v V output buffer (Q8_1Block*)
         * @param bias_q Q bias tensor (optional)
         * @param bias_k K bias tensor (optional)
         * @param bias_v V bias tensor (optional)
         * @param m Batch size (rows)
         * @param n_q Q output columns
         * @param n_k K output columns
         * @param k Input/weight inner dimension
         * @param k_block_size Block size for K quantization
         * @return true on success
         */
        virtual bool execute_q8_1(
            const Q8_1Block *input_q8_1,
            void *output_q,
            void *output_k,
            void *output_v,
            const TensorBase *bias_q,
            const TensorBase *bias_k,
            const TensorBase *bias_v,
            int m, int n_q, int n_k, int k,
            int k_block_size = 64) = 0;

        /**
         * @brief Execute fused QKV GEMM with Q8_1 input and uniform Q8_1 output
         *
         * @param input_q8_1 Pre-quantized input [m, k] as Q8_1 blocks
         * @param output_q Q output buffer (Q8_1Block*)
         * @param output_k K output buffer (Q8_1Block*)
         * @param output_v V output buffer (Q8_1Block*)
         * @param bias_q Q bias tensor (optional)
         * @param bias_k K bias tensor (optional)
         * @param bias_v V bias tensor (optional)
         * @param m Batch size (rows)
         * @param n_q Q output columns
         * @param n_k K output columns
         * @param k Input/weight inner dimension
         * @return true on success
         */
        virtual bool execute_q8_1_to_q8_1(
            const Q8_1Block *input_q8_1,
            void *output_q,
            void *output_k,
            void *output_v,
            const TensorBase *bias_q,
            const TensorBase *bias_k,
            const TensorBase *bias_v,
            int m, int n_q, int n_k, int k) = 0;
    };

    /**
     * @brief Rotary Positional Embedding (RoPE) kernel
     */
    class ITensorRoPE : public ITensorKernel
    {
    public:
        /**
         * @brief Apply RoPE to Q8_1 input, output to FP32 (Hybrid mode)
         *
         * This method dequantizes Q8_1 input, applies rotation, and outputs
         * to FP32 WITHOUT requantization. Used in Hybrid activation precision
         * mode to eliminate the dequant→rotate→requant cycle.
         *
         * @param Q_in Q8_1 Q input tensor [seq_len, n_heads * head_dim]
         * @param K_in Q8_1 K input tensor [seq_len, n_kv_heads * head_dim] or nullptr
         * @param Q_out FP32 Q output tensor [seq_len, n_heads * head_dim]
         * @param K_out FP32 K output tensor [seq_len, n_kv_heads * head_dim] or nullptr
         * @param position_ids Position indices [seq_len]
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of KV heads
         * @param head_dim Head dimension
         * @param rope_theta RoPE frequency base
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index
         * @return true on success
         */
        virtual bool apply_q8_1_to_fp32(
            TensorBase *Q_in,
            TensorBase *K_in,
            TensorBase *Q_out,
            TensorBase *K_out,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)Q_in;
            (void)K_in;
            (void)Q_out;
            (void)K_out;
            (void)position_ids;
            (void)seq_len;
            (void)n_heads;
            (void)n_kv_heads;
            (void)head_dim;
            (void)rope_theta;
            (void)mpi_ctx;
            (void)device_idx;
            return false; // Default: not supported
        }

        /**
         * @brief Apply RoPE to Q8_1 input, output to Q16_1 (HybridQ16 mode)
         *
         * This method dequantizes Q8_1 input, applies rotation, and outputs
         * to Q16_1 (higher precision). Used in HybridQ16 activation precision
         * mode for the Q16 fused attention kernel.
         *
         * Q16_1 output has 256× finer precision than Q8_1, enabling high-accuracy
         * integer attention without FP32 intermediate storage.
         *
         * @param Q_in Q8_1 Q input tensor [seq_len, n_heads * head_dim]
         * @param K_in Q8_1 K input tensor [seq_len, n_kv_heads * head_dim] or nullptr
         * @param Q_out Q16_1 Q output tensor [seq_len, n_heads * head_dim]
         * @param K_out Q16_1 K output tensor [seq_len, n_kv_heads * head_dim] or nullptr
         * @param position_ids Position indices [seq_len]
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of KV heads
         * @param head_dim Head dimension
         * @param rope_theta RoPE frequency base
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index
         * @return true on success
         */
        virtual bool apply_q8_1_to_q16_1(
            TensorBase *Q_in,
            TensorBase *K_in,
            TensorBase *Q_out,
            TensorBase *K_out,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)Q_in;
            (void)K_in;
            (void)Q_out;
            (void)K_out;
            (void)position_ids;
            (void)seq_len;
            (void)n_heads;
            (void)n_kv_heads;
            (void)head_dim;
            (void)rope_theta;
            (void)mpi_ctx;
            (void)device_idx;
            return false; // Default: not supported
        }

        /**
         * @brief Apply RoPE to Q8_1 input, output to Q16 with per-head scale output
         *
         * Pure integer arithmetic implementation: Q8 fixed-point ratios, Q15 sin/cos,
         * integer rotation without FP32 intermediate storage. Only FP32 used is for
         * computing the output scale.
         *
         * @param Q_in Q8_1 Q input tensor [seq_len, n_heads * head_dim]
         * @param K_in Q8_1 K input tensor [seq_len, n_kv_heads * head_dim] or nullptr
         * @param Q_out Q16 Q output tensor [seq_len, n_heads * head_dim]
         * @param K_out Q16 K output tensor [seq_len, n_kv_heads * head_dim] or nullptr
         * @param Q_head_scales Output: per-head Q scales [seq_len * n_heads] or nullptr
         * @param K_head_scales Output: per-head K scales [seq_len * n_kv_heads] or nullptr
         * @param block_size Q16 output block size (32, 64, 128, or 192)
         * @param position_ids Position indices [seq_len]
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of KV heads
         * @param head_dim Head dimension
         * @param rope_theta RoPE frequency base
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index
         * @return true on success
         */
        virtual bool apply_q8_1_to_q16(
            TensorBase *Q_in,
            TensorBase *K_in,
            TensorBase *Q_out,
            TensorBase *K_out,
            float *Q_head_scales,
            float *K_head_scales,
            Q16BlockSize block_size,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)Q_in;
            (void)K_in;
            (void)Q_out;
            (void)K_out;
            (void)Q_head_scales;
            (void)K_head_scales;
            (void)block_size;
            (void)position_ids;
            (void)seq_len;
            (void)n_heads;
            (void)n_kv_heads;
            (void)head_dim;
            (void)rope_theta;
            (void)mpi_ctx;
            (void)device_idx;
            return false; // Default: not supported
        }

        /**
         * @brief Apply RoPE to Q8_1 input, output to Q16 with FIXED scale
         *
         * Pure integer arithmetic implementation with fixed output scale.
         * All output Q16 blocks have d = kv_cache_scale / 32767, matching
         * the KV cache quantization scale. This enables true integer attention
         * where Q, K, and V all use identical scales.
         *
         * Algorithm:
         * 1. Compute per-block scale ratios (O(head_dim/32) FP32 ops)
         * 2. Rescale Q8 to fixed scale (pure integer per-element)
         * 3. Apply RoPE rotation (pure integer)
         * 4. Pack to Q16 with fixed d = kv_cache_scale / 32767
         *
         * @param Q_in Q8_1 Q input tensor [seq_len, n_heads * head_dim]
         * @param K_in Q8_1 K input tensor [seq_len, n_kv_heads * head_dim] or nullptr
         * @param Q_out Q16 Q output tensor [seq_len, n_heads * head_dim]
         * @param K_out Q16 K output tensor [seq_len, n_kv_heads * head_dim] or nullptr
         * @param block_size Q16 output block size (32, 64, 128, or 192)
         * @param position_ids Position indices [seq_len]
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of KV heads
         * @param head_dim Head dimension
         * @param rope_theta RoPE frequency base
         * @param kv_cache_scale Fixed scale for all output blocks (e.g., 8.0f)
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index
         * @return true on success
         */
        virtual bool apply_q8_1_to_q16_fixed_scale(
            TensorBase *Q_in,
            TensorBase *K_in,
            TensorBase *Q_out,
            TensorBase *K_out,
            Q16BlockSize block_size,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            float kv_cache_scale,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)Q_in;
            (void)K_in;
            (void)Q_out;
            (void)K_out;
            (void)block_size;
            (void)position_ids;
            (void)seq_len;
            (void)n_heads;
            (void)n_kv_heads;
            (void)head_dim;
            (void)rope_theta;
            (void)kv_cache_scale;
            (void)mpi_ctx;
            (void)device_idx;
            return false; // Default: not supported
        }

        /**
         * @brief Apply RoPE to Q16_1 input, output Q16_1 with dynamic per-head scale
         *
         * This method takes Q16_1 input (with per-block scales from GEMM) and applies
         * RoPE rotation, outputting Q16_1 with potentially different per-head scales.
         * This is used for K projections in HybridQ16 mode where GEMM outputs K as Q16_1
         * to preserve precision (256× finer than Q8_1).
         *
         * For Q: K_in is processed in-place (Q8_1→Q16_1 with fixed scale, same as before)
         * For K: K_in is already Q16_1 from GEMM, apply dynamic-scale RoPE
         *
         * Dynamic Scale Algorithm:
         * 1. Dequantize input Q16_1 to FP32 (per-block scales from GEMM)
         * 2. Apply RoPE rotation (FP32)
         * 3. Find max_abs per head (determines output scale)
         * 4. Requantize to Q16_1 with per-head scale
         * 5. Store per-head scales in K_head_scales for attention
         *
         * @param Q_in Q8_1 Q input tensor [seq_len, n_heads * head_dim]
         * @param K_in Q16_1 K input tensor [seq_len, n_kv_heads * head_dim] (from GEMM!)
         * @param Q_out Q16_1 Q output tensor [seq_len, n_heads * head_dim]
         * @param K_out Q16_1 K output tensor [seq_len, n_kv_heads * head_dim]
         * @param K_head_scales Output: per-head K scales [seq_len * n_kv_heads]
         * @param block_size Q16 block size (64 or 128, matching head_dim)
         * @param position_ids Position indices [seq_len]
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of KV heads
         * @param head_dim Head dimension
         * @param rope_theta RoPE frequency base
         * @param q_kv_cache_scale Fixed scale for Q output (same as V/KV cache)
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index
         * @return true on success
         */
        virtual bool apply_mixed_q8_k16_to_q16(
            TensorBase *Q_in,     // Q8_1 input for Q
            TensorBase *K_in,     // Q16_1 input for K (from GEMM!)
            TensorBase *Q_out,    // Q16_1 output for Q
            TensorBase *K_out,    // Q16_1 output for K
            float *K_head_scales, // Output: per-head K scales [seq_len * n_kv_heads]
            Q16BlockSize block_size,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            float q_kv_cache_scale, // Fixed scale for Q output
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)Q_in;
            (void)K_in;
            (void)Q_out;
            (void)K_out;
            (void)K_head_scales;
            (void)block_size;
            (void)position_ids;
            (void)seq_len;
            (void)n_heads;
            (void)n_kv_heads;
            (void)head_dim;
            (void)rope_theta;
            (void)q_kv_cache_scale;
            (void)mpi_ctx;
            (void)device_idx;
            return false; // Default: not supported
        }

        /**
         * @brief Apply RoPE using tensor objects with automatic type dispatch
         *
         * Inspects Q/K tensor native_type() and dispatches to the appropriate
         * typed method (apply, apply_bf16, apply_fp16, apply_q8_1).
         *
         * @param Q Query tensor [seq_len, n_heads * head_dim] - modified in-place
         * @param K Key tensor [seq_len, n_kv_heads * head_dim] - modified in-place
         * @param position_ids Position indices [seq_len]
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of KV heads
         * @param head_dim Head dimension
         * @param rope_theta RoPE frequency base
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index
         * @return true on success, false on failure or unsupported type
         *
         * @note Default returns false. Subclasses should override with type-aware dispatch.
         *
         * @note If position_ids is nullptr, positions are computed as (pos_offset + seq_idx)
         *       on the GPU, eliminating the need for host-to-device memory copy.
         *       This is the PREFERRED path for contiguous positions (decode and most prefill).
         *
         * @note This is the PRIMARY RoPE interface. All implementations must provide this.
         *
         * @param rotary_dim Number of dimensions to rotate per head. When 0 (default),
         *        rotates all head_dim dimensions (full RoPE). When < head_dim, only the
         *        first rotary_dim dimensions are rotated (partial RoPE); the remaining
         *        dimensions are left untouched. The stride between heads remains head_dim.
         *        Used by models like Qwen 3.5 where partial_rotary_factor < 1.0.
         */
        virtual bool apply_tensor(
            TensorBase *Q,
            TensorBase *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            int pos_offset = 0,
            int rotary_dim = 0) = 0;

        /**
         * @brief Apply RoPE to MTP verifier rows using serial-decode-equivalent math.
         *
         * Production MTP verifier graphs process a tiny M=2..4 row block, but
         * the accepted-state publication contract is defined by serial decode:
         * row 0, then row 1, and so on, each at its absolute position.  Backend
         * implementations should provide a grouped kernel entry point for that
         * contract rather than making `RoPEStage` allocate one-row tensors and
         * replay itself.
         *
         * @return true when the backend applied the grouped verifier contract.
         *         The default is fail-closed so callers cannot silently fall
         *         back to non-equivalent prefill math.
         */
        virtual bool apply_verifier_rows_decode_equivalent(
            TensorBase *Q,
            TensorBase *K,
            const int *position_ids,
            int verifier_rows,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            int pos_offset = 0,
            int rotary_dim = 0)
        {
            (void)Q;
            (void)K;
            (void)position_ids;
            (void)verifier_rows;
            (void)n_heads;
            (void)n_kv_heads;
            (void)head_dim;
            (void)rope_theta;
            (void)mpi_ctx;
            (void)device_idx;
            (void)pos_offset;
            (void)rotary_dim;
            return false;
        }

        /**
         * @brief Update the pos_offset stored in device params for graph replay
         *
         * GPU implementations should use this as a pre-capture/pre-replay hook:
         * upload the new value to a workspace-owned device buffer on the explicit
         * stage stream. Captured RoPE execution must not record H2D memcpy nodes.
         *
         * @param pos_offset New position offset for the next decode step
         */
        virtual void setDynamicPosOffset(int pos_offset)
        {
            (void)pos_offset;
        }

        /**
         * @brief Pre-upload explicit position IDs for graph replay.
         *
         * Request-batched speculative sidecars can have rows such as `[N, N]`
         * because each row belongs to a different logical request.  Those rows
         * cannot use the contiguous `pos_offset + row` device-param path.  GPU
         * implementations should copy the host position IDs into their
         * workspace-owned position buffer on the already-bound explicit stream
         * before graph capture/replay starts.  Captured RoPE execution must
         * launch kernels only; it must not record host-to-device copies.
         *
         * @param position_ids Host position IDs for the next replay.
         * @param seq_len Number of valid entries in `position_ids`.
         */
        virtual void setDynamicPositionIds(const int *position_ids, int seq_len)
        {
            (void)position_ids;
            (void)seq_len;
        }

        /**
         * @brief Bind explicit position IDs that already live on the device.
         *
         * This is the no-copy counterpart to setDynamicPositionIds().  Resident
         * MTP continuation publishes per-request position rows in device
         * workspace memory; GPU RoPE implementations should read this pointer
         * directly during graph capture/replay.  Implementations must reject
         * null/default streams before treating the pointer as valid so the
         * producer/consumer ordering can be expressed with explicit events.
         *
         * @param position_ids_device Device pointer to INT32 position IDs.
         * @param seq_len Number of valid entries in `position_ids_device`.
         */
        virtual void setDynamicDevicePositionIds(const void *position_ids_device, int seq_len)
        {
            (void)position_ids_device;
            (void)seq_len;
        }
    };

    /**
     * @brief SwiGLU activation kernel
     */
    class ITensorSwiGLU : public ITensorKernel
    {
    public:
        /**
         * @brief Apply SwiGLU using tensor objects with automatic type dispatch
         *
         * Inspects gate/up/output tensor native_type() and dispatches to the appropriate
         * typed method (apply, apply_bf16, apply_fp16, apply_q8_1).
         *
         * @param gate Gate tensor [rows, cols] (read-only)
         * @param up Up tensor [rows, cols] (read-only)
         * @param output Output tensor [rows, cols]
         * @param rows Number of rows
         * @param cols Number of columns
         * @param add_residual Whether to add residual
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index
         * @return true on success, false on failure or unsupported type combination
         *
         * @note Default returns false. Subclasses should override with type-aware dispatch.
         *
         * @note This is the PRIMARY SwiGLU interface. All implementations must provide this.
         */
        virtual bool apply_tensor(
            const TensorBase *gate,
            const TensorBase *up,
            TensorBase *output,
            int rows, int cols,
            bool add_residual,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;
    };

    /**
     * @brief Residual Add kernel interface
     *
     * Element-wise addition of input and residual tensors.
     * Supports FP32, BF16, FP16 with automatic type dispatch via apply_tensor().
     */
    class ITensorResidualAdd : public ITensorKernel
    {
    public:
        /**
         * @brief Apply residual add using tensor objects with automatic type dispatch
         *
         * Inspects input/residual/output tensor native_type() and dispatches to
         * the appropriate typed method (apply, apply_bf16, apply_fp16).
         *
         * @param input Input tensor
         * @param residual Residual tensor
         * @param output Output tensor
         * @param num_elements Number of elements to process
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index
         * @return true on success, false on failure or unsupported type
         *
         * @note This is the PRIMARY ResidualAdd interface. All implementations must provide this.
         */
        virtual bool apply_tensor(
            const TensorBase *input,
            const TensorBase *residual,
            TensorBase *output,
            size_t num_elements,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;
    };

    /**
     * @brief Embedding lookup kernel
     *
     * Handles embedding table lookup with typed output:
     * - FP32 output: Direct memcpy from embedding table
     * - BF16/FP16 output: Lookup FP32 embeddings, then convert
     * - Q8_1 output: Lookup FP32 embeddings, then quantize to Q8_1 format
     *
     * The embedding table is always FP32 (stored in model weights).
     * Output format is determined by activation precision configuration.
     */
    class ITensorEmbedding : public ITensorKernel
    {
    public:
        /**
         * @brief Apply embedding lookup using tensor objects with automatic type dispatch
         *
         * @param embed_table Embedding table tensor [vocab_size, d_model] (FP32)
         * @param token_ids Token IDs to look up
         * @param num_tokens Number of tokens
         * @param d_model Embedding dimension
         * @param output Output tensor [num_tokens, d_model]
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index
         * @return true on success, false on failure or unsupported type
         *
         * @note This is the PRIMARY Embedding interface. All implementations must provide this.
         */
        virtual bool apply_tensor(
            const TensorBase *embed_table,
            const int *token_ids,
            int num_tokens,
            int d_model,
            TensorBase *output,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;

        /**
         * @brief Provide a model-owned prepared embedding handle for execution.
         *
         * Graph-built model paths resolve this through PreparedWeightStore.
         * Implementations may keep using their legacy lookup/fallback path when
         * no handle is provided (for direct kernel tests and non-model callers).
         */
        virtual void setPreparedEmbeddingHandle(const PreparedEmbeddingHandle *handle)
        {
            (void)handle;
        }

        /**
         * @brief Update token IDs stored in a graph-safe device-side buffer for replay
         *
         * GPU embedding kernels can override this to copy token IDs from the caller's
         * host buffer into pinned host memory and then into the stable device workspace
         * buffer captured by the decode graph. CPU implementations can keep the default
         * no-op behavior.
         */
        virtual void setDynamicTokenIds(const int *token_ids, int num_tokens)
        {
            (void)token_ids;
            (void)num_tokens;
        }

        /**
         * @brief Use token IDs that already live in device memory.
         *
         * GPU graph paths call this when a previous device-side sampler produced
         * the next token into an arena/workspace buffer. Implementations should
         * make the next apply_tensor() consume `token_ids_device` directly on the
         * bound explicit stream, without uploading host token IDs or touching the
         * CUDA/HIP null stream. CPU kernels keep the default no-op and continue to
         * use setDynamicTokenIds()/apply_tensor() host pointers.
         *
         * @param token_ids_device Device pointer to INT32 token IDs [num_tokens].
         * @param num_tokens Number of token IDs available at the device pointer.
         */
        virtual void setDynamicDeviceTokenIds(const void *token_ids_device, int num_tokens)
        {
            (void)token_ids_device;
            (void)num_tokens;
        }

        /**
         * @brief Set local vocab range for vocab-parallel embedding shards.
         *
         * Local TP runs multiple devices inside one MPI rank, so MPI rank cannot
         * identify the vocabulary shard. CPU implementations use this range to
         * zero tokens outside the local shard before the embedding allreduce.
         */
        virtual void setVocabRange(int vocab_offset, int local_vocab_size)
        {
            (void)vocab_offset;
            (void)local_vocab_size;
        }

        /**
         * @brief Declare whether tokens outside the local vocab range are legal.
         *
         * Vocab-parallel embeddings intentionally return zero rows for tokens
         * owned by another TP participant. A later allreduce sums the one
         * non-zero shard back into the replicated hidden state. GPU kernels use
         * this flag only for host/device-token validation; the actual embedding
         * kernels still enforce the local range and write zeros for misses.
         */
        virtual void setAllowOutOfRangeTokenIds(bool allow)
        {
            (void)allow;
        }
    };

    /**
     * @brief Softmax kernel
     *
     * Computes row-wise softmax normalization with optional causal masking.
     * Typically used for attention score normalization.
     *
     * The modern interface uses apply_tensor() for polymorphic dispatch.
     * Legacy typed methods (apply_bf16, apply_fp16) are retained for
     * backward compatibility but implementations should prefer apply_tensor().
     */
    class ITensorSoftmax : public ITensorKernel
    {
    public:
        /**
         * @brief Apply softmax using tensor objects with automatic type dispatch
         *
         * Modern polymorphic interface. Inspects input/output tensor native_type()
         * and dispatches to the appropriate precision-specific implementation.
         *
         * @param input Input tensor [rows, cols] (any supported precision)
         * @param output Output tensor [rows, cols] (same type as input, may be same buffer for in-place)
         * @param rows Number of rows
         * @param cols Number of columns per row
         * @param use_causal_mask Apply causal (lower-triangular) masking
         * @param scale Pre-softmax scale factor (typically 1/sqrt(d_k) for attention)
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index (-1 = CPU)
         * @return true on success, false on failure or unsupported type
         *
         * @note This is the PRIMARY Softmax interface. All implementations must provide this.
         */
        virtual bool apply_tensor(
            const TensorBase *input,
            TensorBase *output,
            int rows, int cols,
            bool use_causal_mask,
            float scale = 1.0f,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;
    };

    /**
     * @brief RMS Normalization kernel
     */
    class ITensorRMSNorm : public ITensorKernel
    {
    public:
        /**
         * @brief Apply RMSNorm using tensor objects with automatic type dispatch
         *
         * Inspects input/output tensor native_type() and dispatches to the appropriate
         * typed method (apply, apply_bf16, apply_fp16, apply_q8_1).
         *
         * @param input Input tensor [rows, cols]
         * @param weight Weight tensor (gamma) [cols] - always FP32
         * @param output Output tensor [rows, cols] - must match input type
         * @param rows Number of rows
         * @param cols Number of columns
         * @param epsilon RMSNorm epsilon
         * @param mpi_ctx MPI context (optional)
         * @param device_idx Device index
         * @return true on success, false on failure or unsupported type combination
         *
         * @note This is the PRIMARY RMSNorm interface. All implementations must provide this.
         */
        virtual bool apply_tensor(
            const TensorBase *input,
            const TensorBase *weight,
            TensorBase *output,
            int rows, int cols,
            float epsilon = 1e-6f,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1) = 0;

        /**
         * @brief Execute fused RMSNorm + INT8 quantization (operator fusion API)
         *
         * Fuses RMS normalization with per-row INT8 quantization in a single pass.
         * Eliminates intermediate FP32 buffer and redundant memory traffic.
         *
         * Algorithm:
         * 1. Compute RMS per row: rms = sqrt(mean(x²) + eps)
         * 2. Normalize: x_norm = x / rms
         * 3. Apply gamma: x_scaled = x_norm * gamma
         * 4. Quantize per-row: x_int8 = round(x_scaled / scale), scale = max(|x_scaled|) / 127
         *
         * Performance Benefits:
         * - Saves 1 FP32 intermediate buffer allocation (rows × cols × 4 bytes)
         * - Reduces memory bandwidth by ~33% (1 write instead of 2)
         * - Improves cache efficiency (single-pass algorithm)
         * - Expected speedup: 5-10% per RMSNorm operation
         *
         * @param input Input tensor [rows, cols] FP32
         * @param weight RMSNorm scale parameters [cols] FP32 (gamma)
         * @param output Output tensor [rows, cols] INT8
         * @param scales Per-row quantization scales [rows] FP32 (output parameter)
         * @param rows Number of rows (sequence length)
         * @param cols Hidden dimension (d_model)
         * @param epsilon RMSNorm epsilon for numerical stability (default: 1e-6)
         * @param mpi_ctx MPI context for distributed execution (nullptr = single node)
         * @param device_idx Device index for execution (-1 = host/CPU, ≥0 = GPU)
         *
         * @return true on success, false if not supported by this kernel
         *
         * @note Default implementation returns false (not implemented).
         *       FusedRMSNormQuantize kernel overrides this for optimized fused path.
         *       Standard RMSNorm kernels do not implement this (use apply() instead).
         */
        virtual bool execute(
            const float *input,
            const float *weight,
            int8_t *output,
            float *scales,
            int rows,
            int cols,
            float epsilon = 1e-6f,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)input;
            (void)weight;
            (void)output;
            (void)scales;
            (void)rows;
            (void)cols;
            (void)epsilon;
            (void)mpi_ctx;
            (void)device_idx;
            return false; // Not implemented by default (use apply() for standard RMSNorm)
        }
    };

    // =========================================================================
    // GDN (Gated Delta Network) Kernel Interfaces
    // =========================================================================

    /**
     * @brief Interface for short causal depthwise conv1d + activation
     *
     * Used by GDN layers for the QKV preprocessing convolution.
     * Implementations: CPUShortConvolution (Phase C), CUDAShortConvolution (Phase F)
     *
     * @see ShortConv1dStage for the compute stage that delegates to this interface
     */
    class ITensorShortConvolution
    {
    public:
        virtual ~ITensorShortConvolution() = default;

        /// Set the GPU stream for kernel dispatch (no-op for CPU implementations)
        virtual void setGPUStream(void *stream) { (void)stream; }

        /// Allocate GPU state buffer (no-op for CPU implementations)
        virtual void allocateGPUState(int state_size) { (void)state_size; }

        /// Allocate GPU scratch used by graph/stage execution (no-op for CPU implementations)
        virtual bool allocateGPUScratch(int scratch_size)
        {
            (void)scratch_size;
            return true;
        }

        /**
         * @brief Bind a caller-owned scratch buffer for in-place prefill.
         *
         * GPU implementations use this shared workspace instead of allocating
         * one persistent scratch buffer per layer. Passing nullptr unbinds the
         * shared buffer and restores the implementation's fallback behavior.
         *
         * @param scratch Device pointer to [max_seq_len * channels] floats.
         * @param scratch_size Number of float elements available in scratch.
         */
        virtual void bindScratchWorkspace(float *scratch, int scratch_size)
        {
            (void)scratch;
            (void)scratch_size;
        }

        /**
         * @brief Bind workspace for verifier-row conv-state snapshots.
         *
         * When enabled for an all-position MTP verifier graph, implementations may
         * write the conv state after each candidate verifier row into this buffer.
         * Layout is [rows, channels, kernel_size - 1] in FP32.
         */
        virtual void bindVerifierStateCaptureWorkspace(float *workspace, int rows, int state_size)
        {
            (void)workspace;
            (void)rows;
            (void)state_size;
        }

        /**
         * @brief Bind workspace for speculative verifier conv state.
         *
         * Phase 13.8 verifier forwards must not mutate the live request conv
         * state before the accepted-count publication step. GPU implementations
         * use this caller-owned buffer as the mutable state for speculative
         * verifier rows after copying the live state into it on the explicit
         * graph/replay stream. Passing nullptr disables the speculative
         * workspace and must cause verifier-state capture paths to fail rather
         * than falling back to live-state mutation.
         */
        virtual void bindSpeculativeStateWorkspace(float *workspace, int state_size)
        {
            (void)workspace;
            (void)state_size;
        }

        /**
         * @brief Return true if this kernel can isolate live conv state per request.
         *
         * Batched MTP verifier publication is only correct when every request
         * has its own live conv-state slot.  A single shared state pointer would
         * let request N's verifier rows become request N+1's initial state.
         */
        virtual bool supportsRequestLiveStateBank(int request_count, int state_size) const
        {
            (void)request_count;
            (void)state_size;
            return false;
        }

        /**
         * @brief Restore a previously captured verifier-row conv state.
         *
         * CPU implementations copy into dst_state. GPU implementations may ignore
         * dst_state and restore their implementation-owned device state.
         */
        virtual bool restoreVerifierStateCaptureRow(float *dst_state, int row, void *stream)
        {
            (void)dst_state;
            (void)row;
            (void)stream;
            return false;
        }

        /**
         * @brief Restore a captured verifier-row conv state by device row index.
         *
         * GPU MTP publication uses compact device metadata to choose the
         * accepted verifier row. Implementations that support this method must
         * read @p device_row_index on @p stream and copy the corresponding
         * snapshot into implementation-owned live state without any host read.
         * @p dst_state is retained for the scalar restore ABI shape but must
         * not be refreshed by device-indexed GPU implementations; host mirror
         * adoption must be a separate explicit operation. The stream is
         * intentionally mandatory for GPU implementations.
         */
        virtual bool restoreVerifierStateCaptureRowFromDeviceIndex(
            float *dst_state,
            const int *device_row_index,
            void *stream)
        {
            (void)dst_state;
            (void)device_row_index;
            (void)stream;
            return false;
        }

        /**
         * @brief Restore request-batched conv state from device row indices.
         *
         * `device_row_indices` names one accepted verifier row per request.
         * Implementations must restore into request-owned live-state storage
         * laid out by the backend, not into one shared `gpu_state_` buffer. A
         * negative row index means no row was accepted for that request and the
         * corresponding live-state slot must remain unchanged. The optional
         * `dst_states` mirror is retained for ABI symmetry but must not be
         * refreshed by GPU hot-path implementations.
         */
        virtual bool restoreVerifierStateCaptureRowsFromDeviceIndices(
            float *dst_states,
            int dst_state_stride_floats,
            const int *device_row_indices,
            int request_count,
            int row_index_stride,
            void *stream)
        {
            (void)dst_states;
            (void)dst_state_stride_floats;
            (void)device_row_indices;
            (void)request_count;
            (void)row_index_stride;
            (void)stream;
            return false;
        }

        /// Return true when padded prefill can commit state using a dynamic real length.
        virtual bool supportsPaddedPrefillRealLength() const { return false; }

        /// Reset GPU state to zero (no-op for CPU implementations)
        virtual void resetGPUState() {}

        /// Size of implementation-owned recurrent state, if exportable.
        virtual size_t stateBytes() const { return 0; }

        /// Export implementation-owned state. When stream is non-null, GPU
        /// implementations may enqueue async copies; callers must synchronize
        /// the stream before consuming the exported payload.
        virtual bool exportState(void *dst_host, void *dst_device, void *stream) const
        {
            (void)dst_host;
            (void)dst_device;
            (void)stream;
            return stateBytes() == 0;
        }

        /// Import implementation-owned state. When stream is non-null, GPU
        /// implementations may enqueue async copies; callers must synchronize
        /// the stream before using the imported state.
        virtual bool importState(const void *src_host, const void *src_device, void *stream)
        {
            (void)src_host;
            (void)src_device;
            (void)stream;
            return stateBytes() == 0;
        }

        /**
         * @brief Apply causal depthwise conv1d + optional SiLU activation
         *
         * Handles both prefill (seq_len > 1) and decode (seq_len == 1):
         * - Prefill: full causal conv1d over sequence with zero-padding
         * - Decode: single-step update using conv_state history
         *
         * @param input     Input data [seq_len, channels]
         * @param weight    Conv weight [channels, kernel_size]
         * @param bias      Optional bias [channels] (nullptr = no bias)
         * @param output    Output [seq_len, channels]
         * @param conv_state Conv state [channels, kernel_size-1] for decode continuity
         *                   (nullptr allowed for prefill-only, required for decode)
         * @param seq_len   Sequence length (1 = decode mode)
         * @param channels  Number of channels (depthwise)
         * @param kernel_size Convolution kernel width
         * @param apply_silu Whether to apply SiLU activation after convolution
         * @return true on success
         */
        virtual bool forward(
            const float *input, const float *weight, const float *bias,
            float *output, float *conv_state,
            int seq_len, int channels, int kernel_size,
            bool apply_silu = true) = 0;

        /**
         * @brief Apply short conv and snapshot the mutable conv state after each row.
         *
         * Snapshot rows are laid out as:
         *   state_snapshots[row * snapshot_stride_floats + state_index]
         * where state_index spans [channels, kernel_size - 1]. The caller may
         * restore one row to avoid replaying accepted verifier tokens.
         */
        virtual bool forwardWithStateSnapshots(
            const float *input, const float *weight, const float *bias,
            float *output, float *conv_state,
            int seq_len, int channels, int kernel_size,
            float *state_snapshots, int snapshot_stride_floats,
            int max_snapshot_rows,
            bool apply_silu = true)
        {
            (void)input;
            (void)weight;
            (void)bias;
            (void)output;
            (void)conv_state;
            (void)seq_len;
            (void)channels;
            (void)kernel_size;
            (void)state_snapshots;
            (void)snapshot_stride_floats;
            (void)max_snapshot_rows;
            (void)apply_silu;
            return false;
        }

        /**
         * @brief Restore conv state from a row produced by forwardWithStateSnapshots().
         */
        virtual bool restoreStateFromSnapshot(
            float *state, const float *state_snapshots,
            int snapshot_row, int snapshot_stride_floats,
            int state_floats, void *stream = nullptr)
        {
            (void)state;
            (void)state_snapshots;
            (void)snapshot_row;
            (void)snapshot_stride_floats;
            (void)state_floats;
            (void)stream;
            return false;
        }

        /**
         * @brief Prefill variant whose state commit length is read from device memory.
         *
         * GPU graph replay reads the current real length from device memory.
         * Stages upload that scalar before capture/replay on an explicit stream;
         * captured stage bodies must not record H2D nodes. CPU implementations
         * normally do not use this entry point; stages pass the effective length
         * directly to forward().
         */
        virtual bool forwardWithEffectiveSeqLen(
            const float *input, const float *weight, const float *bias,
            float *output, float *conv_state,
            int seq_len, int channels, int kernel_size,
            const int *device_effective_seq_len,
            bool apply_silu = true)
        {
            (void)input;
            (void)weight;
            (void)bias;
            (void)output;
            (void)conv_state;
            (void)seq_len;
            (void)channels;
            (void)kernel_size;
            (void)device_effective_seq_len;
            (void)apply_silu;
            return false;
        }

        /**
         * @brief Request-batched conv forward with request-owned live state.
         *
         * `seq_len` is the flattened row count, laid out as
         * `[request_count, request_seq_len, channels]`. Implementations must
         * update request-local live-state slots for ordinary decode/prefill and
         * must use request-local speculative slots when verifier capture is
         * active. Snapshot rows remain in the existing flat verifier namespace,
         * so request `r`, row `t` writes slot `r * request_seq_len + t`.
         */
        virtual bool forwardBatchedRequests(
            const float *input, const float *weight, const float *bias,
            float *output, float *conv_state,
            int seq_len, int request_count, int request_seq_len,
            int channels, int kernel_size,
            bool apply_silu = true)
        {
            (void)input;
            (void)weight;
            (void)bias;
            (void)output;
            (void)conv_state;
            (void)seq_len;
            (void)request_count;
            (void)request_seq_len;
            (void)channels;
            (void)kernel_size;
            (void)apply_silu;
            return false;
        }
    };

    /**
     * @brief Interface for Gated Delta Net recurrence kernel
     *
     * Implements the delta rule linear attention recurrence:
     *   S_t = exp(g_t) * S_{t-1}
     *   kv = S_t * k_t
     *   delta = (v_t - kv) * beta_t
     *   S_t += outer(k_t, delta)
     *   o_t = S_t * q_t
     *
     * Inputs are pre-processed: Q/K are already L2-normalized and scaled,
     * gate/beta are already computed (exp-decay gate, sigmoid-beta).
     *
     * Implementations: CPUGatedDeltaNet (Phase C), CUDAGatedDeltaNet (Phase F)
     *
     * @see GDNRecurrenceStage for the compute stage that delegates to this interface
     */
    class ITensorGatedDeltaNet
    {
    public:
        virtual ~ITensorGatedDeltaNet() = default;

        /// Set the GPU stream for kernel dispatch (no-op for CPU implementations)
        virtual void setGPUStream(void *stream) { (void)stream; }

        /// Allocate GPU state buffer (no-op for CPU implementations)
        virtual void allocateGPUState(int state_size) { (void)state_size; }

        /// Reset GPU state to zero (no-op for CPU implementations)
        virtual void resetGPUState() {}

        /// Size of implementation-owned recurrent state, if exportable.
        virtual size_t stateBytes() const { return 0; }

        /// Export implementation-owned state. When stream is non-null, GPU
        /// implementations may enqueue async copies; callers must synchronize
        /// the stream before consuming the exported payload.
        virtual bool exportState(void *dst_host, void *dst_device, void *stream) const
        {
            (void)dst_host;
            (void)dst_device;
            (void)stream;
            return stateBytes() == 0;
        }

        /// Import implementation-owned state. When stream is non-null, GPU
        /// implementations may enqueue async copies; callers must synchronize
        /// the stream before using the imported state.
        virtual bool importState(const void *src_host, const void *src_device, void *stream)
        {
            (void)src_host;
            (void)src_device;
            (void)stream;
            return stateBytes() == 0;
        }

        /// Check if GPU state is allocated with the required size.
        /// Used by graph capture preflight to verify warmup has occurred.
        virtual bool isGPUStateReady(int required_state_size) const
        {
            (void)required_state_size;
            return false;
        }

        /// Return true when padded prefill can commit recurrence state using a dynamic real length.
        virtual bool supportsPaddedPrefillRealLength() const { return false; }

        /**
         * @brief Bind workspace for verifier-row recurrence-state snapshots.
         *
         * When enabled for an all-position MTP verifier graph, implementations may
         * write the recurrence state after each candidate verifier row into this
         * buffer. Layout is [rows, n_heads, d_k, d_v] in FP32.
         */
        virtual void bindVerifierStateCaptureWorkspace(float *workspace, int rows, int state_size)
        {
            (void)workspace;
            (void)rows;
            (void)state_size;
        }

        /**
         * @brief Bind workspace for speculative verifier recurrence state.
         *
         * Stateful GPU verifier forwards use this caller-owned buffer as the
         * mutable recurrence state while writing verifier state slots.
         */
        virtual void bindSpeculativeStateWorkspace(float *workspace, int state_size)
        {
            (void)workspace;
            (void)state_size;
        }

        /**
         * @brief Return true if this kernel can isolate recurrence state per request.
         *
         * Request-batched verifier rows need one live recurrence state slot per
         * request. This capability deliberately excludes scalar loops over a
         * single shared state pointer.
         */
        virtual bool supportsRequestLiveStateBank(int request_count, int state_size) const
        {
            (void)request_count;
            (void)state_size;
            return false;
        }

        /**
         * @brief Restore a previously captured verifier-row recurrence state.
         *
         * CPU implementations copy into dst_state. GPU implementations may ignore
         * dst_state and restore their implementation-owned device state.
         */
        virtual bool restoreVerifierStateCaptureRow(float *dst_state, int row, void *stream)
        {
            (void)dst_state;
            (void)row;
            (void)stream;
            return false;
        }

        /**
         * @brief Restore a captured verifier-row recurrence state by device row index.
         *
         * This is the device-resident companion to restoreVerifierStateCaptureRow().
         * It exists so MTP accepted-state publication can consume compact GPU
         * verifier metadata directly instead of synchronizing a row index to
         * the host. GPU implementations must use @p stream, must not launch
         * on an implicit/default stream, and must not refresh @p dst_state from
         * device memory. Host mirror adoption belongs in a separate explicit
         * non-hot-path operation.
         */
        virtual bool restoreVerifierStateCaptureRowFromDeviceIndex(
            float *dst_state,
            const int *device_row_index,
            void *stream)
        {
            (void)dst_state;
            (void)device_row_index;
            (void)stream;
            return false;
        }

        /**
         * @brief Restore request-batched recurrence state from device row indices.
         *
         * This is the tensor-kernel companion to
         * IComputeStage::restoreVerifierStateCaptureRowsFromDeviceIndices().
         * GPU implementations must consume @p device_row_indices on @p stream
         * and write each request into independent live-state storage. A
         * negative row index means that request stays at its pre-verifier
         * state. Returning false is the correct behavior until both the capture
         * layout and the live-state bank are request-aware.
         */
        virtual bool restoreVerifierStateCaptureRowsFromDeviceIndices(
            float *dst_states,
            int dst_state_stride_floats,
            const int *device_row_indices,
            int request_count,
            int row_index_stride,
            void *stream)
        {
            (void)dst_states;
            (void)dst_state_stride_floats;
            (void)device_row_indices;
            (void)request_count;
            (void)row_index_stride;
            (void)stream;
            return false;
        }

        /**
         * @brief Deinterleave merged QKV buffer on device (GPU-only)
         *
         * Splits a merged [seq_len, q_dim + k_dim + v_dim] device buffer into
         * separate contiguous Q, K, V device arrays. Handles Qwen GDN modular
         * Q/K tiling:
         *   k_head_for_v_head_j = (j + global_v_head_offset) % n_k_heads
         *
         * GPU implementations allocate persistent grow-only scratch internally.
         * CPU implementations return false (deinterleave done on host by stage).
         *
         * @param d_merged_qkv  Device pointer to merged QKV buffer
         * @param d_q           [out] Device pointer to deinterleaved Q [seq_len, n_v_heads * d_k]
         * @param d_k           [out] Device pointer to deinterleaved K [seq_len, n_v_heads * d_k]
         * @param d_v           [out] Device pointer to deinterleaved V [seq_len, n_v_heads * d_v]
         * @param seq_len       Sequence length
         * @param n_k_heads     Key head count in merged buffer
         * @param n_v_heads     Value head count (output head count)
         * @param d_k           Key/query head dimension
         * @param d_v           Value head dimension
         * @param global_v_head_offset  TP modular repeat offset
         * @return true if device deinterleave succeeded, false for CPU fallback
         */
        virtual bool deinterleave_qkv_device(
            const float *d_merged_qkv,
            float *&d_q, float *&d_k, float *&d_v,
            int seq_len, int n_k_heads, int n_v_heads,
            int head_dim_k, int head_dim_v, int global_v_head_offset)
        {
            (void)d_merged_qkv;
            (void)d_q;
            (void)d_k;
            (void)d_v;
            (void)seq_len;
            (void)n_k_heads;
            (void)n_v_heads;
            (void)head_dim_k;
            (void)head_dim_v;
            (void)global_v_head_offset;
            return false;
        }

        /**
         * @brief Bind caller-owned scratch for merged-QKV deinterleaving.
         *
         * GPU implementations use this workspace instead of keeping a private
         * grow-only deinterleave buffer per GDN layer. Passing nullptr unbinds
         * the shared buffer and restores the implementation's fallback behavior.
         *
         * @param scratch Device pointer to the deinterleave scratch buffer.
         * @param scratch_size Number of float elements available in scratch.
         */
        virtual void bindDeinterleaveWorkspace(float *scratch, size_t scratch_size)
        {
            (void)scratch;
            (void)scratch_size;
        }

        /**
         * @brief Chunk-parallel prefill: process full sequence
         *
         * The kernel is responsible for ALL math: L2 normalization,
         * query scaling, gate computation, and recurrence.
         *
         * @param Q         Raw query [seq_len, n_heads * d_k]
         * @param K         Raw key [seq_len, n_heads * d_k]
         * @param V         Value [seq_len, n_heads * d_v]
         * @param alpha     A projection [seq_len, n_heads]
         * @param beta_raw  B projection [seq_len, n_heads]
         * @param A_log     Learnable log-space gate [n_heads]
         * @param dt_bias   Learnable dt bias [n_heads]
         * @param output    Output context [seq_len, n_heads * d_v]
         * @param state     Recurrence state [n_heads, d_k, d_v] (updated in-place)
         * @param seq_len   Sequence length
         * @param n_heads   Number of attention heads
         * @param d_k       Key head dimension
         * @param d_v       Value head dimension
         * @param chunk_size Chunk size for chunked processing (0 = sequential)
         * @param use_qk_l2norm Apply L2 normalization to Q and K
         * @return true on success
         */
        virtual bool chunk_forward(
            const float *Q, const float *K, const float *V,
            const float *alpha, const float *beta_raw,
            const float *A_log, const float *dt_bias,
            float *output, float *state,
            int seq_len, int n_heads, int d_k, int d_v,
            int chunk_size, bool use_qk_l2norm) = 0;

        /**
         * @brief Chunk prefill with recurrence-state snapshots after each row.
         *
         * Snapshot rows are laid out as:
         *   state_snapshots[row * snapshot_stride_floats + state_index]
         * where state_index spans [n_heads, d_k, d_v].
         */
        virtual bool chunkForwardWithStateSnapshots(
            const float *Q, const float *K, const float *V,
            const float *alpha, const float *beta_raw,
            const float *A_log, const float *dt_bias,
            float *output, float *state,
            int seq_len, int n_heads, int d_k, int d_v,
            int chunk_size, bool use_qk_l2norm,
            float *state_snapshots, int snapshot_stride_floats,
            int max_snapshot_rows)
        {
            (void)Q;
            (void)K;
            (void)V;
            (void)alpha;
            (void)beta_raw;
            (void)A_log;
            (void)dt_bias;
            (void)output;
            (void)state;
            (void)seq_len;
            (void)n_heads;
            (void)d_k;
            (void)d_v;
            (void)chunk_size;
            (void)use_qk_l2norm;
            (void)state_snapshots;
            (void)snapshot_stride_floats;
            (void)max_snapshot_rows;
            return false;
        }

        /**
         * @brief Decode-equivalent verifier chunk directly from a merged QKV row layout.
         *
         * Some graph builders store GDN Q/K/V as one tensor row:
         *
         *   [Q(n_k_heads*d_k) | K(n_k_heads*d_k) | V(n_heads*d_v)]
         *
         * The generic stage path can deinterleave that tensor into separate
         * contiguous Q/K/V matrices before calling chunkForwardWithStateSnapshots(),
         * but all-position MTP verifier chunks are tiny (M=2..4), so the copy can
         * dominate CPU replay time.  Implementations that can read this merged
         * layout directly should override this method and publish the same
         * post-row snapshots as serial recurrent_step().
         *
         * The Q/K head for local V-head h is:
         *
         *   qk_head = (h + global_v_head_offset) mod n_k_heads
         *
         * Returning false means the implementation has no direct merged-QKV
         * verifier path; callers may use their explicit deinterleave path when
         * that behavior is intended.
         */
        virtual bool chunkForwardMergedQKVWithStateSnapshots(
            const float *merged_qkv, int qkv_stride,
            const float *alpha, const float *beta_raw,
            const float *A_log, const float *dt_bias,
            float *output, float *state,
            int seq_len, int n_k_heads, int n_heads, int d_k, int d_v,
            int global_v_head_offset, int chunk_size, bool use_qk_l2norm,
            float *state_snapshots, int snapshot_stride_floats,
            int max_snapshot_rows)
        {
            (void)merged_qkv;
            (void)qkv_stride;
            (void)alpha;
            (void)beta_raw;
            (void)A_log;
            (void)dt_bias;
            (void)output;
            (void)state;
            (void)seq_len;
            (void)n_k_heads;
            (void)n_heads;
            (void)d_k;
            (void)d_v;
            (void)global_v_head_offset;
            (void)chunk_size;
            (void)use_qk_l2norm;
            (void)state_snapshots;
            (void)snapshot_stride_floats;
            (void)max_snapshot_rows;
            return false;
        }

        /**
         * @brief Restore recurrence state from a row produced by chunkForwardWithStateSnapshots().
         */
        virtual bool restoreStateFromSnapshot(
            float *state, const float *state_snapshots,
            int snapshot_row, int snapshot_stride_floats,
            int state_floats, void *stream = nullptr)
        {
            (void)state;
            (void)state_snapshots;
            (void)snapshot_row;
            (void)snapshot_stride_floats;
            (void)state_floats;
            (void)stream;
            return false;
        }

        /**
         * @brief Chunk prefill variant with a graph-replay-updated real length.
         *
         * The launch shape remains tied to seq_len (the bucket size), while the
         * kernel reads device_effective_seq_len to decide how many leading rows
         * advance recurrent state. Padded output rows are inert and must not
         * affect subsequent decode state.
         */
        virtual bool chunkForwardWithEffectiveSeqLen(
            const float *Q, const float *K, const float *V,
            const float *alpha, const float *beta_raw,
            const float *A_log, const float *dt_bias,
            float *output, float *state,
            int seq_len, int n_heads, int d_k, int d_v,
            int chunk_size, bool use_qk_l2norm,
            const int *device_effective_seq_len)
        {
            (void)Q;
            (void)K;
            (void)V;
            (void)alpha;
            (void)beta_raw;
            (void)A_log;
            (void)dt_bias;
            (void)output;
            (void)state;
            (void)seq_len;
            (void)n_heads;
            (void)d_k;
            (void)d_v;
            (void)chunk_size;
            (void)use_qk_l2norm;
            (void)device_effective_seq_len;
            return false;
        }

        /**
         * @brief Request-batched GDN recurrence with request-owned live state.
         *
         * `seq_len` is flattened as `[request_count, request_seq_len]`.
         * Implementations must process each request against its own live or
         * speculative recurrence-state slot and write snapshots into the flat
         * verifier-row namespace (`request * request_seq_len + row`).
         */
        virtual bool chunkForwardBatchedRequests(
            const float *Q, const float *K, const float *V,
            const float *alpha, const float *beta_raw,
            const float *A_log, const float *dt_bias,
            float *output, float *state,
            int seq_len, int request_count, int request_seq_len,
            int n_heads, int d_k, int d_v,
            int chunk_size, bool use_qk_l2norm)
        {
            (void)Q;
            (void)K;
            (void)V;
            (void)alpha;
            (void)beta_raw;
            (void)A_log;
            (void)dt_bias;
            (void)output;
            (void)state;
            (void)seq_len;
            (void)request_count;
            (void)request_seq_len;
            (void)n_heads;
            (void)d_k;
            (void)d_v;
            (void)chunk_size;
            (void)use_qk_l2norm;
            return false;
        }

        /**
         * @brief Single-step decode recurrence
         *
         * The kernel is responsible for ALL math: L2 normalization,
         * query scaling, gate computation, and recurrence.
         *
         * @param q         Raw query [n_heads * d_k]
         * @param k         Raw key [n_heads * d_k]
         * @param v         Value [n_heads * d_v]
         * @param alpha     A projection [n_heads]
         * @param beta_raw  B projection [n_heads]
         * @param A_log     Learnable log-space gate [n_heads]
         * @param dt_bias   Learnable dt bias [n_heads]
         * @param output    Output context [n_heads * d_v]
         * @param state     Recurrence state [n_heads, d_k, d_v] (updated in-place)
         * @param n_heads   Number of attention heads
         * @param d_k       Key head dimension
         * @param d_v       Value head dimension
         * @param use_qk_l2norm Apply L2 normalization to Q and K
         * @return true on success
         */
        virtual bool recurrent_step(
            const float *q, const float *k, const float *v,
            const float *alpha, const float *beta_raw,
            const float *A_log, const float *dt_bias,
            float *output, float *state,
            int n_heads, int d_k, int d_v,
            bool use_qk_l2norm) = 0;
    };

} // namespace llaminar2
