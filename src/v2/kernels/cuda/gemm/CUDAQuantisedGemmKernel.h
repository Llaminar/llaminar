/**
 * @file CUDAQuantisedGemmKernel.h
 * @brief CUDA quantized GEMM kernel for quantized tensors
 *
 * Implements ITensorGemm using CUDA quantized GEMM backends for any quantized weight tensor.
 * This is the CUDA counterpart to CPU QuantisedGemmKernel.
 *
 * **Design**:
 * - Primary entry point: multiply_tensor() with type introspection
 * - Supports any quantized weight type (IQ4_NL, Q8_0, Q4_0, Q4_K, etc.)
 * - Weights are packed once and cached with preferred and active execution families
 * - NativeVNNI-native formats keep compact payloads as the active execution family while
 *   still retaining an Int8Expanded fallback mirror in the cache
 * - Activations quantized on-the-fly or used directly if already Q8_1
 * - Uses CUTLASS INT8 Tensor Core GEMM (SM 8.0+ Ampere)
 *
 * **Type Dispatch** (in multiply_tensor):
 * | A type | C type | Path |
 * |--------|--------|------|
 * | Q8_1   | FP32   | INT8×INT8→FP32 |
 * | Q8_1   | Q8_1   | INT8×INT8→Q8_1 (fused requant) |
 * | FP32   | FP32   | quant A→INT8×INT8→FP32 |
 * | FP32   | Q8_1   | quant A→INT8×INT8→Q8_1 |
 *
 * **Usage**:
 * ```cpp
 * auto kernel = std::make_unique<CUDAQuantisedGemmKernel>(weights, cuda_device_id);
 * kernel->multiply_tensor(activations, output, ...);
 * ```
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "CUDAWeightPacker.h"
#include "tensors/TensorKernels.h"
#include "tensors/BlockStructures.h"
#include "interfaces/IWorkspaceConsumer.h"
#include <memory>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

namespace llaminar2
{
    // Forward declarations
    class TensorBase;
    class Q8_1Tensor;
    class FP32Tensor;
    class WeightPreloader; // For friend class access to ensureWeightsConverted

    namespace cuda
    {

        /**
         * @brief CUDA GEMM kernel for quantized weight tensors using CUDA packed weights
         *
         * Implements ITensorGemm for any quantized weight tensor type.
         *
         * **Supported Weight Types**:
         * - IQ4_NLTensor
         * - Q8_0Tensor
         * - Q4_0Tensor
         * - Q4_KTensor, Q5_KTensor, Q6_KTensor
         * - Any tensor implementing dequantization
         *
         * **Compute Path**:
         * 1. Weights: Pack once, recording preferred and active CUDA weight families
         * 2. Current execution: active_family selects the primary path; NativeVNNI-native
         *    formats still retain an Int8Expanded fallback mirror for fallback/tuning paths
         * 3. Activations: Q8_1 blocks used directly, or FP32→INT8 quantized
         * 4. GEMM: CUTLASS INT8×INT8→INT32 (Tensor Core mma.sync.m16n8k32)
         * 5. Output: INT32 × scale_A × scale_B → FP32 (or requant to Q8_1)
         *
         * **Memory Layout**:
         * - Int8Expanded fallback weights: INT8 [K × N] ColumnMajor (CUTLASS Tensor Core requirement)
         * - Activations: INT8 [M × K] RowMajor
         * - Output: FP32/INT32 [M × N] RowMajor
         *
         * **Performance**:
         * - Weight conversion: Once per tensor (cached)
         * - CUTLASS Tensor Core: 50-90 TFLOPS on RTX 3090
         * - Activation quantization: Per-row symmetric, fused with GEMM launch
         */
        struct CUDAConcurrentPrefillPool; // Forward declaration (definition in .cpp)

        class CUDAQuantisedGemmKernel : public ITensorGemm, public IWorkspaceConsumer
        {
        public:
            struct Impl; // Forward declaration for PIMPL (definition in .cpp)

            // Legacy stubs — NativeVNNI is always enabled; CUTLASS fallback removed.
            // Kept for ABI compatibility with test code; will be removed in a future cleanup.
            static void setNativeVNNIEnabled(bool enabled);
            static bool isNativeVNNIEnabled();
            static void setForceCutlassFallback(bool enabled);
            static bool isForceCutlassFallback();

            /**
             * @brief Release all per-device shared CUDAConcurrentPrefillPool
             *        singletons (CUDA streams, events, and scratch buffers).
             *
             * The pool is shared across kernel instances on the same device, so it
             * must be torn down when KernelFactory::clearCache() runs to avoid
             * leaking static state between test runs / orchestrator resets.
             */
            static void clearSharedPrefillPools();

            /**
             * @brief Release input-dependent CUDA execution contexts.
             *
             * Keeps packed weights resident, but drops per-session GEMV, prefill,
             * and cuBLAS scratch/handles so a new request cannot inherit state
             * from the previous prompt shape or capture stream.
             */
            void resetDynamicState() override;

            /// @brief Returns true when any CUDA execution context is live.
            bool hasDynamicStateActive() const override;

            /**
             * @brief Construct kernel for quantized weight tensor (lazy conversion)
             *
             * @param weights Any quantized tensor (must be on GPU)
             * @param cuda_device_id CUDA device ID (from cudaGetDevice, NOT global index)
             *
             * @throws std::runtime_error if weight not quantized or not on GPU
             */
            CUDAQuantisedGemmKernel(const TensorBase *weights, int cuda_device_id);

            /**
             * @brief Construct kernel from pre-packed CUDA weights (preferred path)
             *
             * This constructor avoids redundant weight conversion by using packed
             * CUDAPackedWeights that are cached in the tensor's cache_ field.
             *
             * @param packed Pre-packed CUDA weights from KernelFactory cache
             * @param cuda_device_id CUDA device ID
             *
             * @throws std::runtime_error if packed is null or has invalid dimensions
             */
            CUDAQuantisedGemmKernel(CUDAPackedWeights *packed, int cuda_device_id);

            /**
             * @brief Construct kernel from pre-uploaded device pointers (MoE batch path)
             *
             * Used for MoE expert weights that are batch-packed and uploaded as a single
             * contiguous allocation. The kernel references device pointers at calculated
             * offsets into the shared allocation.
             *
             * @param N Output features (rows per expert)
             * @param K Input features (columns)
             * @param cuda_device_id CUDA device ID
             * @param d_vnni Device pointer to NativeVNNI payload for this expert
             * @param d_scales Device pointer to FP16 scales for this expert
             * @param d_mins Device pointer to FP16 mins (nullptr if symmetric)
             * @param d_emins Device pointer to extended mins (nullptr if not needed)
             * @param codebook_id NativeVNNI codebook identifier
             * @param blocks_per_row Number of 32-element blocks per row (K/32)
             * @param lifetime_owner Shared pointer that keeps the GPU allocation alive
             */
            CUDAQuantisedGemmKernel(
                int N, int K, int cuda_device_id,
                uint8_t *d_vnni, uint16_t *d_scales, uint16_t *d_mins, uint32_t *d_emins,
                uint8_t codebook_id, uint32_t blocks_per_row,
                std::shared_ptr<void> lifetime_owner);

            ~CUDAQuantisedGemmKernel() override;

            // Non-copyable
            CUDAQuantisedGemmKernel(const CUDAQuantisedGemmKernel &) = delete;
            CUDAQuantisedGemmKernel &operator=(const CUDAQuantisedGemmKernel &) = delete;

            // Movable
            CUDAQuantisedGemmKernel(CUDAQuantisedGemmKernel &&) noexcept;
            CUDAQuantisedGemmKernel &operator=(CUDAQuantisedGemmKernel &&) noexcept;

            // =========================================================================
            // ITensorGemm interface - Primary entry points
            // =========================================================================

            /**
             * @brief Tensor-based GEMM with type introspection (PRIMARY ENTRY POINT)
             *
             * Inspects A->native_type() and C->native_type() to select optimal path:
             * - Q8_1 → Q8_1: Zero-copy INT8 GEMM
             * - Q8_1 → FP32: INT8 GEMM + dequant
             * - FP32 → FP32: Quantize A + INT8 GEMM + dequant
             * - FP32 → Q8_1: Quantize A + INT8 GEMM + requant
             *
             * @param A Input activations tensor [m, k] (Q8_1 or FP32)
             * @param C Output tensor [m, n] (Q8_1 or FP32)
             * @param transpose_B Whether weights are transposed (ignored, always true)
             * @param alpha Scale factor for result
             * @param beta Scale for existing C (accumulate if != 0)
             * @param bias Optional bias tensor [n] to add after GEMM (nullptr = no bias)
             * @param mpi_ctx MPI context (unused for CUDA kernel)
             * @param device_idx Device index (unused, kernel bound to cuda_device_id_)
             * @param workspace Pre-allocated device workspace (nullptr = kernel allocates)
             *
             * @return true on success
             */
            bool multiply_tensor(
                const TensorBase *A, TensorBase *C,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const TensorBase *bias = nullptr,
                const IMPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                DeviceWorkspaceManager *workspace = nullptr,
                int activation_row_offset = 0) override;

            /**
             * @brief Tensor-based GEMM with explicit dimensions
             *
             * Same as above but with explicit m, n, k for pre-allocated buffers.
             */
            bool multiply_tensor(
                const TensorBase *A, TensorBase *C,
                int m, int n, int k,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const TensorBase *bias = nullptr,
                const IMPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                DeviceWorkspaceManager *workspace = nullptr,
                int activation_row_offset = 0) override;

            /**
             * @brief Tensor-aware fused multi-projection GEMM
             *
             * Optimized implementation that:
             * 1. Ensures input is on GPU (uploads if needed)
             * 2. Quantizes input to INT8 ONCE
             * 3. Runs all projections with the same quantized input
             * 4. Marks output tensors as device-dirty
             *
             * This is the preferred path for FusedQKVGEMMStage on GPU.
             *
             * @param input Input tensor [m, k] (will be uploaded to GPU if needed)
             * @param projections Vector of TensorProjectionDesc (kernels + output tensors)
             * @param m Number of rows (seq_len)
             * @param k Input dimension (d_model)
             * @param mpi_ctx MPI context (unused)
             * @param workspace Pre-allocated device workspace (nullptr = kernel allocates)
             * @return true on success
             */
            bool multiply_fused_tensor(
                const TensorBase *input,
                const std::vector<TensorProjectionDesc> &projections,
                int m, int k,
                const IMPIContext *mpi_ctx = nullptr,
                DeviceWorkspaceManager *workspace = nullptr) override;

            /**
             * @brief Activation-activation GEMM (not supported for quantized kernel)
             *
             * CUDAQuantisedGemmKernel is for weight projections only.
             * For activation-activation GEMM, use a dedicated attention kernel.
             */
            bool multiply_activations(
                const float *A, const float *B, float *C,
                int m, int n, int k,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const IMPIContext *mpi_ctx = nullptr,
                int device_idx = -1);

            /**
             * @brief Strided activation-activation GEMM (not supported)
             */
            bool multiply_activations_strided(
                const float *A, const float *B, float *C,
                int m, int n, int k,
                int lda, int ldb, int ldc,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const IMPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            // =========================================================================
            // ITensorKernel interface
            // =========================================================================

            bool supports_device(int device_idx) const override;

            void setGPUStream(void *stream) override { gpu_stream_ = stream; }

            // =========================================================================
            // IKernelSnapshotCapable interface
            // =========================================================================

            KernelSnapshotInfo getKernelSnapshotInfo() const override;

            // =========================================================================
            // IWorkspaceConsumer Interface
            // =========================================================================

            /**
             * @brief Get workspace requirements for GEMM at given dimensions
             *
             * Returns buffers needed for INT8 GEMM:
             * - quant_a [M×K]: INT8 quantized activations
             * - scales_a [M]: per-row activation scales
             * - acc_int32 [M×N]: INT32 GEMM accumulator
             *
             * @param m Maximum sequence length (batch size)
             * @param n Number of output features (0 = use internal N_)
             * @param k Number of input features (0 = use internal K_)
             * @return WorkspaceRequirements describing all needed buffers
             *
             * @note Call with maximum expected M to allocate once and reuse.
             */
            WorkspaceRequirements getWorkspaceRequirements(
                int m, int n = 0, int k = 0) const override;

            /**
             * @brief Bind workspace manager for managed mode
             *
             * After binding, the kernel uses pre-allocated buffers from the
             * workspace manager instead of internal allocations.
             *
             * @param workspace Pointer to workspace manager (NOT owned, must outlive kernel)
             *                  Pass nullptr to return to legacy mode.
             */
            void bindWorkspace(DeviceWorkspaceManager *workspace) override;

            /**
             * @brief Check if a workspace is currently bound
             */
            bool hasWorkspace() const override;

            /**
             * @brief Get the currently bound workspace manager
             */
            DeviceWorkspaceManager *getWorkspace() const override;

            // =========================================================================
            // Accessors
            // =========================================================================

            int cuda_device_id() const { return cuda_device_id_; }
            int getCudaDeviceId() const { return cuda_device_id_; }
            void *getGPUStream() const { return gpu_stream_; }

            size_t weight_rows() const { return N_; }
            size_t weight_cols() const { return K_; }
            bool weights_converted() const { return weights_converted_; }

            /// @brief Export native-VNNI device pointers for grouped MoE CUDA prefill.
            bool exportNativeVNNIMatrixDesc(DeviceNativeVNNIMatrixDesc &out) override;

            /**
             * @brief Prepare weights for efficient execution (ITensorGemm interface)
             *
             * For CUDA: converts weights to INT8 + uploads to device memory.
             * Call this during weight preloading to avoid first-use overhead.
             */
            void prepareWeights() override { ensureWeightsConverted(); }

            // =========================================================================
            // Fused Activation+GEMM entry points
            // =========================================================================

            /**
             * @brief GEMM with fused SwiGLU activation preprocessing
             *
             * Replaces the standalone SwiGLU kernel + activation quantization with
             * a single fused kernel: silu(gate) * up → INT8, then runs GEMM.
             * Eliminates one FP32 write + read of the intermediate SwiGLU output.
             *
             * @param d_gate  Gate activations [m, k] FP32 on device
             * @param d_up    Up activations [m, k] FP32 on device
             * @param d_C     Output [m, n] FP32 on device
             * @param m       Rows (sequence length)
             * @param n       Output columns
             * @param k       Input columns (intermediate_size)
             * @param alpha   Output scale
             * @param beta    Accumulation scale (0 = overwrite)
             * @return true on success
             */
            bool multiply_with_fused_swiglu(
                const float *d_gate, const float *d_up,
                float *d_C,
                int m, int n, int k,
                float alpha = 1.0f, float beta = 0.0f);

            /**
             * @brief Tensor-based GEMM with fused SwiGLU activation
             *
             * High-level entry point for GEMMStage. Handles GPU coherence:
             * extracts device pointers from gate/up tensors, runs fused
             * SwiGLU+quantize+GEMM, marks output as device-dirty.
             *
             * @param gate   Gate tensor [m, k] (must be on GPU)
             * @param up     Up tensor [m, k] (must be on GPU)
             * @param output Output tensor [m, n]
             * @param m      Rows
             * @param n      Output columns
             * @param k      Input columns
             * @param alpha  Output scale
             * @param beta   Accumulation scale
             * @return true on success
             */
            bool multiply_tensor_with_fused_swiglu(
                const TensorBase *gate, const TensorBase *up,
                TensorBase *output,
                int m, int n, int k,
                float alpha = 1.0f, float beta = 0.0f);

        private:
            // =========================================================================
            // Internal dispatch methods
            // =========================================================================

            /**
             * @brief Q8_1 activations → INT8 GEMM → FP32 output
             */
            bool multiply_q8_to_fp32(
                const Q8_1Block *d_A_q8, float *d_C,
                int m, int n, int k,
                float alpha, float beta);

            /**
             * @brief Q8_1 activations → INT8 GEMM → Q8_1 output (fused requant)
             */
            bool multiply_q8_to_q8(
                const Q8_1Block *d_A_q8, Q8_1Block *d_C_q8,
                int m, int n, int k);

            /**
             * @brief FP32 activations → quantize → INT8 GEMM → FP32 output
             */
            bool multiply_fp32_to_fp32(
                const float *d_A, float *d_C,
                int m, int n, int k,
                float alpha, float beta);

            /**
             * @brief FP32 activations → quantize → INT8 GEMM → FP32 output + bias
             * @param d_bias Optional bias vector [N], broadcasted across rows
             */
            bool multiply_fp32_to_fp32_with_bias(
                const float *d_A, float *d_C, const float *d_bias,
                int m, int n, int k,
                float alpha, float beta);

            /**
             * @brief Small-M FP32 activations -> native-VNNI row-wise GEMV.
             *
             * Greedy MTP verifier forwards commonly use M=2. The generic native
             * VNNI prefill path is correct for that shape, but it is much slower
             * than the decode-class GEMV path. This helper quantizes all rows
             * once, then dispatches the tuned M=1 GEMV kernel per row on the
             * caller-provided graph stream.
             */
            bool multiply_fp32_to_fp32_small_m_gemv(
                const float *d_A, float *d_C, const float *d_bias,
                int m, int n, int k,
                float alpha, float beta);

            /**
             * @brief FP32 activations → quantize → INT8 GEMM → Q8_1 output
             */
            bool multiply_fp32_to_q8(
                const float *d_A, Q8_1Block *d_C_q8,
                int m, int n, int k);

            // =========================================================================
            // Internal impl methods (for workspace override)
            // =========================================================================

            /**
             * @brief Implementation for multiply_tensor with current workspace_ state
             */
            bool multiply_tensor_impl(
                const TensorBase *A, TensorBase *C,
                int m, int n, int k,
                float alpha, float beta,
                const TensorBase *bias,
                int activation_row_offset = 0);

            /**
             * @brief Implementation for multiply_fused_tensor with current workspace_ state
             */
            bool multiply_fused_tensor_impl(
                const TensorBase *input,
                const std::vector<TensorProjectionDesc> &projections,
                int m, int k);

            // =========================================================================
            // Weight conversion
            // =========================================================================

            /**
             * @brief Ensure weights are converted to INT8 + scales
             *
             * Converts from native quantized format to:
             * - d_weights_int8_: INT8 [K × N] ColumnMajor
             * - d_scales_: float [N] per-column scales
             *
             * Conversion is done once and cached.
             */
            void ensureWeightsConverted();

            /**
             * @brief Validate workspace is bound and has required buffers
             *
             * @throws std::runtime_error if workspace not bound (kernels require workspace)
             */
            void validateWorkspace() const;

            // =========================================================================
            // Member data
            // =========================================================================

            const TensorBase *weights_ = nullptr;  // Original weight tensor (null if using packed_)
            CUDAPackedWeights *packed_ = nullptr;  // Pre-packed weights (owned by tensor cache)
            std::shared_ptr<void> lifetime_owner_; // Keeps shared MoE batch allocation alive
            int cuda_device_id_;
            size_t N_; // Output features (weight rows)
            size_t K_; // Input features (weight cols)

            // Converted INT8 weight representation (cached) - only used with legacy constructor
            // When packed_ is set, these are unused (data comes from packed_->d_int8_data/d_scales)
            int8_t *d_weights_int8_ = nullptr; // [K × N] ColumnMajor
            float *d_scales_B_ = nullptr;      // [N] per-column scales
            bool weights_converted_ = false;
            bool owns_weight_memory_ = false; // true if we allocated d_weights_int8_/d_scales_B_

            // IWorkspaceConsumer state - REQUIRED for execution
            // Kernels do not own any work buffers; all buffers come from workspace
            DeviceWorkspaceManager *workspace_ = nullptr; ///< Bound workspace manager (not owned, REQUIRED)

            // Per-kernel-instance unique slice id for the TEMP_C_FP32 redirect
            // buffer. Each cached GEMM kernel needs its OWN slice in the shared
            // workspace; otherwise a kernel can clobber the redirect source of
            // another kernel whose async D2D copy is still pending. Set in the
            // constructors from a static atomic counter.
            uint32_t slice_id_ = 0;

            // Helper: unique workspace buffer name for this kernel's TEMP_C_FP32 slice.
            std::string tempCFp32BufferName() const
            {
                return std::string(GemmWorkspaceBuffers::TEMP_C_FP32) + "_" + std::to_string(slice_id_);
            }

            // GPU stream for graph capture (nullptr = default stream)
            void *gpu_stream_ = nullptr;

            // Concurrent prefill pool is a per-device shared singleton (see
            // getSharedCUDAPrefillPool in CUDAQuantisedGemmKernel.cpp) to avoid
            // allocating duplicate scratch buffers per kernel instance.

            // PIMPL for CUTLASS implementation (definition in .cpp)
            std::unique_ptr<Impl> impl_;

            // Friend class for WeightPreloader (deprecated, will be removed)
            friend class llaminar2::WeightPreloader;
        };

    } // namespace cuda
} // namespace llaminar2
