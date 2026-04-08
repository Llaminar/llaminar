/**
 * @file CPUNativeVNNIGemmKernel.h
 * @brief ITensorGemm implementation for CPU NativeVNNI GEMM/GEMV.
 *
 * This kernel keeps weights in their native quantized format (Q4_0, IQ4_NL, etc.)
 * and decodes blocks inline during computation using AVX-512 VNNI (vpdpbusd).
 *
 * ## Comparison with CPUQuantisedGemmKernel
 *
 * | Aspect | CPUQuantisedGemmKernel | CPUNativeVNNIGemmKernel |
 * |--------|--------------------|-----------------------|
 * | Weight packing | Decode to INT8 at pack time | Keep native bytes, decode at runtime |
 * | Weight memory | 1 byte/element | 0.5 byte/element (Q4_0) |
 * | GEMV bandwidth | 2× memory traffic | 1× memory traffic |
 * | Decode cost | Zero (pre-decoded) | Small (nibble unpack) |
 * | Best for | M>1 (compute-bound) | M=1 (memory-bound GEMV) |
 *
 * ## Supported Formats (Phase 1)
 *
 * - Q4_0: Simple symmetric 4-bit (16 byte payload / 32 elements)
 * - IQ4_NL: Non-linear 4-bit with LUT (16 byte payload / 32 elements)
 *
 * Additional formats can be added by implementing decode_native_block() cases
 * in CPUNativeVNNIGemv.h.
 */

#pragma once

#include "CPUNativeVNNIWeightPacker.h"
#include "CPUNativeVNNIGemv.h"
#include "tensors/TensorKernels.h"
#include "tensors/TensorClasses.h"
#include "kernels/cpu/primitives/SwiGLUPrimitives.h"
#include "kernels/cpu/rotation/ActivationRotation.h"
#include "utils/Logger.h"

namespace llaminar2::cpu::native_vnni
{

    class CPUNativeVNNIGemmKernel : public ITensorGemm
    {
    public:
        /**
         * @brief Construct from a quantized weight tensor.
         *
         * Packs weights into the CPU NativeVNNI layout at construction time.
         * The tensor must implement IINT8Unpackable and provide vnniFormatInfo().
         *
         * @param weights Source weight tensor [N, K]
         * @param row_start Start row for TP slicing (default 0)
         * @param row_end End row for TP slicing (default -1 = all)
         */
        explicit CPUNativeVNNIGemmKernel(const TensorBase *weights,
                                         int row_start = 0, int row_end = -1)
        {
            // Pick up activation rotation from the weight tensor (if set).
            // When present, activations will be rotated before Q8_1 quantization
            // to reduce kurtosis and improve int8 fidelity.
            // The rotation is also fused into weight packing (dequant→rotate→requant)
            // so that the original tensor format is preserved.
            activation_rotation_ = weights->activationRotation();

            if (!packWeightsCPUNativeVNNI(weights, packed_, row_start, row_end,
                                          activation_rotation_))
            {
                LOG_ERROR("[CPUNativeVNNIGemmKernel] Failed to pack weights");
                valid_ = false;
                return;
            }
            valid_ = true;

            // Store the native block size for this format
            native_block_size_ = native_block_bytes_for_codebook(packed_.codebook_id);

            // ---------------------------------------------------------------
            // Store native blocks for deferred VNNI packing (all formats).
            //
            // This enables memory-efficient operation: instead of keeping the
            // VNNI-interleaved data permanently (~1.1 B/elem overhead), we store
            // only the native quantized blocks (~0.5-1.06 B/elem) and repack
            // into a shared workspace on demand for each GEMM/GEMV call.
            //
            // Superblock formats (Q6_K, Q3_K, etc.) and rotation paths retain
            // permanent interleaved data since repacking from native blocks is
            // not supported (superblocks need full context, rotation fuses into packing).
            // ---------------------------------------------------------------
            const bool is_superblock = packed_.is_superblock;
            const bool can_defer = !is_superblock && !activation_rotation_ && native_block_size_ > 0;

            if (can_defer)
            {
                // Store native blocks from the weight tensor
                storeNativeBlocks(weights, row_start, row_end);
            }

            // Release the permanent interleaved data if deferred packing is active.
            // The workspace will be populated on demand before each GEMM/GEMV call.
            if (can_defer && native_blocks_ptr_ != nullptr)
            {
                packed_.releaseInterleavedData();
                deferred_packing_ = true;

                LOG_DEBUG("[CPUNativeVNNIGemmKernel] Deferred packing: "
                          << packed_.N << "×" << packed_.K
                          << " (codebook=" << (int)packed_.codebook_id
                          << ", block_size=" << native_block_size_
                          << ", owned=" << (!native_blocks_owned_.empty() ? "yes" : "mmap")
                          << ", saved ~" << (interleavedWorkspaceSize(packed_) / (1024 * 1024)) << " MB)");
            }
            else
            {
                LOG_DEBUG("[CPUNativeVNNIGemmKernel] Packed "
                          << packed_.N << "×" << packed_.K
                          << " weights (codebook=" << (int)packed_.codebook_id
                          << ", payload=" << packed_.payload_bytes << " B/block"
                          << ", asymmetric=" << packed_.is_asymmetric
                          << ", rotation=" << (activation_rotation_ != nullptr)
                          << ", permanent_interleaved=true)");
            }
        }

        /**
         * @brief Construct from pre-packed weights (move).
         */
        explicit CPUNativeVNNIGemmKernel(CPUNativeVNNIPackedWeights &&packed)
            : packed_(std::move(packed)), valid_(true) {}

        ~CPUNativeVNNIGemmKernel() override = default;

        // -------------------------------------------------------------------
        // ITensorKernel interface
        // -------------------------------------------------------------------

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only
        }

        // -------------------------------------------------------------------
        // ITensorGemm interface
        // -------------------------------------------------------------------

        /**
         * @brief C[m×n] = A[m×k] @ B_packed[n×k]^T
         *
         * Primary tensor-aware GEMM entry point.
         * For M=1: dispatches to optimized GEMV path.
         * For M>1: dispatches to tiled GEMM path.
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
            int activation_row_offset = 0) override
        {
            (void)transpose_B;
            (void)mpi_ctx;
            (void)workspace;

            if (!valid_ || device_idx != -1)
                return false;

            if (n > packed_.N || k > packed_.K)
            {
                LOG_ERROR("[CPUNativeVNNIGemmKernel] Dimension mismatch: "
                          << "requested n=" << n << " k=" << k
                          << " packed N=" << packed_.N << " K=" << packed_.K);
                return false;
            }

            const float *A_data = A->data() + static_cast<size_t>(activation_row_offset) * k;
            float *C_data = C->mutable_data();

            // Apply activation rotation for kurtosis reduction (if configured)
            A_data = maybe_rotate_activation(A_data, m, k);

            // Handle beta scaling of existing C
            if (beta != 0.0f && beta != 1.0f)
            {
                for (int i = 0; i < m * n; ++i)
                    C_data[i] *= beta;
            }

            // For deferred packing: repack native blocks into shared workspace
            if (deferred_packing_)
            {
                if (m == 1 && packed_.codebook_id == 19)
                    ensureWorkspaceRaw(); // Q8_0 M=1: just memcpy raw blocks (no interleave)
                else
                    ensureWorkspace(); // All other cases: full VNNI interleave
            }

            if (m == 1)
            {
                // Optimized GEMV path
                if (beta == 0.0f && alpha == 1.0f)
                {
                    gemv_native_vnni(packed_, A_data, C_data);
                }
                else
                {
                    // General case: C = alpha * A@B + beta * C
                    std::vector<float> temp(n);
                    gemv_native_vnni(packed_, A_data, temp.data());
                    for (int j = 0; j < n; ++j)
                        C_data[j] += alpha * temp[j];
                }
            }
            else
            {
                // M>1: tiled GEMM
                if (beta == 0.0f && alpha == 1.0f)
                {
                    gemm_native_vnni(packed_, A_data, C_data, m, n);
                }
                else
                {
                    std::vector<float> temp(n);
                    for (int row = 0; row < m; ++row)
                    {
                        gemv_native_vnni(packed_, A_data + row * k, temp.data());
                        for (int j = 0; j < n; ++j)
                            C_data[row * n + j] += alpha * temp[j];
                    }
                }
            }

            // Release workspace after compute
            if (deferred_packing_)
                packed_.clearWorkspace();

            // Apply bias epilogue: C[m, j] += bias[j]
            if (bias)
            {
                const float *bias_data = bias->data();
                apply_bias_epilogue(C_data, bias_data, m, n, n);
            }

            return true;
        }

        // -------------------------------------------------------------------
        // Accessors
        // -------------------------------------------------------------------

        bool isValid() const { return valid_; }

        const CPUNativeVNNIPackedWeights &packedWeights() const { return packed_; }

        uint8_t codebookId() const { return packed_.codebook_id; }

        int get_n() const override { return packed_.N; }
        int get_k() const override { return packed_.K; }

        // -------------------------------------------------------------------
        // Fused SwiGLU + GEMM: output = W @ (silu(gate) * up)
        // -------------------------------------------------------------------

        /**
         * @brief Fused SwiGLU activation + GEMM on CPU.
         *
         * Computes: output = W_down @ (silu(gate) * up)
         * SwiGLU is applied to the input BEFORE quantization and GEMM,
         * which is mathematically correct (gate and up share dimension K,
         * while output has dimension N ≠ K).
         */
        bool multiply_tensor_with_fused_swiglu(
            const TensorBase *gate,
            const TensorBase *up,
            TensorBase *output,
            int m, int n, int k,
            float alpha = 1.0f, float beta = 0.0f) override
        {
            if (!valid_)
                return false;

            const float *gate_fp32 = gate->data();
            const float *up_fp32 = up->data();
            float *output_fp32 = output->mutable_data();

            // Apply SwiGLU to get the GEMM input: temp = silu(gate) * up  [m, k]
            const size_t input_size = static_cast<size_t>(m) * k;

            // Reuse cached scratch buffer to avoid allocation on every decode token
            const size_t needed = input_size;
            if (swiglu_scratch_.size() < needed)
                swiglu_scratch_.resize(needed);

            primitives::compute_swiglu(gate_fp32, up_fp32, swiglu_scratch_.data(),
                                       static_cast<int>(input_size));

            // Apply activation rotation for kurtosis reduction (if configured)
            const float *gemm_input = maybe_rotate_activation(swiglu_scratch_.data(), m, k);

            // Prepare workspace for deferred packing if needed
            if (deferred_packing_)
            {
                if (m == 1 && packed_.codebook_id == 19)
                    ensureWorkspaceRaw();
                else
                    ensureWorkspace();
            }

            // M=1 fast path: call GEMV directly with raw pointer, skip TensorBase wrapper
            if (m == 1 && alpha == 1.0f && beta == 0.0f)
            {
                gemv_native_vnni(packed_, gemm_input, output_fp32);
                if (deferred_packing_)
                    packed_.clearWorkspace();
                return true;
            }

            // M>1 path: call GEMM directly with raw pointer
            if (beta != 0.0f && beta != 1.0f)
            {
                for (int i = 0; i < m * n; ++i)
                    output_fp32[i] *= beta;
            }
            if (beta == 0.0f && alpha == 1.0f)
            {
                gemm_native_vnni(packed_, gemm_input, output_fp32, m, n);
            }
            else
            {
                std::vector<float> temp(n);
                for (int row = 0; row < m; ++row)
                {
                    gemv_native_vnni(packed_, gemm_input + row * k, temp.data());
                    for (int j = 0; j < n; ++j)
                        output_fp32[row * n + j] += alpha * temp[j];
                }
            }
            if (deferred_packing_)
                packed_.clearWorkspace();
            return true;
        }

        // -------------------------------------------------------------------
        // Fused multi-projection with quantize-once + epilogues
        // -------------------------------------------------------------------

        bool supports_fused_projection() const override
        {
            return true;
        }

        bool multiply_fused_tensor(
            const TensorBase *input,
            const std::vector<TensorProjectionDesc> &projections,
            int m, int k,
            const IMPIContext *mpi_ctx = nullptr,
            DeviceWorkspaceManager *workspace = nullptr) override
        {
            if (!valid_)
                return false;

            const float *input_data = input->data();

            // Apply activation rotation for kurtosis reduction (if configured)
            input_data = maybe_rotate_activation(input_data, m, k);

            const int K_blocks = (k + 31) / 32;

            // -----------------------------------------------------------
            // M==1 decode path: delegate to multiply_tensor per projection.
            // multiply_tensor handles ensureWorkspace + gemv + clearWorkspace
            // correctly for all weight formats and configurations.
            // -----------------------------------------------------------
            if (m == 1)
            {
                for (const auto &proj : projections)
                {
                    if (!proj.kernel || !proj.output)
                        return false;
                    
                    bool success = proj.kernel->multiply_tensor(
                        input, proj.output, m, proj.n, k,
                        true, 1.0f, 0.0f, proj.bias, mpi_ctx, -1, workspace);
                    if (!success)
                        return false;
                }
                return true;
            }

            // -----------------------------------------------------------
            // GEMM path (M > 1) or mixed non-VNNI kernels: sequential.
            // GEMM is compute-bound so OMP overhead is negligible.
            // -----------------------------------------------------------
            std::vector<Q8_1Block> shared_q8;
            bool q8_quantized = false;
            auto ensure_q8_quantized = [&]()
            {
                if (q8_quantized)
                    return;
                q8_quantized = true;
                shared_q8.resize(static_cast<size_t>(m) * K_blocks);
                quantize_activations_to_q8_1(input_data, shared_q8.data(), m, k, K_blocks);
            };

            for (const auto &proj : projections)
            {
                if (!proj.kernel || !proj.output)
                    return false;

                float *out_data = proj.output->mutable_data();

                auto *vnni_kernel = dynamic_cast<CPUNativeVNNIGemmKernel *>(proj.kernel);
                if (vnni_kernel && vnni_kernel->valid_)
                {
                    if (vnni_kernel->deferred_packing_)
                        vnni_kernel->ensureWorkspace();

                    ensure_q8_quantized();
                    gemm_native_vnni_preq(vnni_kernel->packed_, shared_q8.data(), out_data, m, proj.n);

                    if (vnni_kernel->deferred_packing_)
                        vnni_kernel->packed_.clearWorkspace();
                }
                else
                {
                    bool success = proj.kernel->multiply_tensor(
                        input, proj.output, m, proj.n, k,
                        true, 1.0f, 0.0f, proj.bias, mpi_ctx, -1, workspace);
                    if (!success)
                        return false;
                    continue;
                }

                if (proj.bias)
                {
                    const float *bias_data = proj.bias->data();
                    apply_bias_epilogue(out_data, bias_data, m, proj.n, proj.n);
                }
            }
            return true;
        }

    private:
        CPUNativeVNNIPackedWeights packed_;
        bool valid_ = false;

        // -------------------------------------------------------------------
        // Native block storage (all formats)
        // -------------------------------------------------------------------

        /// Owned copy of native quantized blocks for all weight formats.
        /// Layout: [N × blocks_per_row × block_size] contiguous bytes.
        /// Used for deferred VNNI repacking.
        std::vector<uint8_t> native_blocks_owned_;

        /// Pointer to native block data — either into native_blocks_owned_
        /// (TP-sliced weights) or into the original mmap region (non-TP views).
        const uint8_t *native_blocks_ptr_ = nullptr;

        /// Size in bytes of a single native quantized block (34 for Q8_0, 18 for Q4_0, etc.)
        size_t native_block_size_ = 0;

        /// Whether deferred packing is active (interleaved data released, repack on demand)
        bool deferred_packing_ = false;

        // Cached scratch buffer for fused SwiGLU+GEMM (avoids malloc per decode token)
        mutable std::vector<float> swiglu_scratch_;

        // Cached Q8_1 quantization buffer for fused projections (avoids malloc per decode token)
        mutable std::vector<Q8_1Block> q8_scratch_;

        // Block-diagonal rotation for activation kurtosis reduction.
        // When set, activations are rotated before Q8_1 quantization for GEMM.
        // The weight must have been pre-rotated with the same rotation.
        const ActivationRotation *activation_rotation_ = nullptr;

        // Cached scratch buffer for rotated activation (avoids malloc per call)
        mutable std::vector<float> rotation_scratch_;

        // -------------------------------------------------------------------
        // Native block storage helper
        // -------------------------------------------------------------------

        /// Store native blocks from the weight tensor.
        /// Uses ITensor::raw_data() + size_bytes() for generic access to any
        /// quantized format's raw block storage — no per-type dynamic_cast needed.
        /// For mmap views: keeps a zero-copy pointer.
        /// For owned data (TP-sliced): copies the raw block bytes.
        void storeNativeBlocks(const TensorBase *weights, int row_start, int row_end)
        {
            if (row_start < 0)
                row_start = 0;
            if (row_end < 0)
                row_end = weights->shape()[0];

            const int total_rows = weights->shape()[0];
            const int N = row_end - row_start;

            // raw_data() + size_bytes() works for all quantized tensor types
            const auto *base = reinterpret_cast<const uint8_t *>(weights->raw_data());
            const size_t total_size = weights->size_bytes();

            if (!base || total_size == 0 || total_rows == 0)
            {
                LOG_WARN("[CPUNativeVNNIGemmKernel] Cannot store native blocks: "
                         << "raw_data()=" << (const void *)base
                         << " size_bytes()=" << total_size
                         << " — keeping permanent interleaved data");
                native_blocks_ptr_ = nullptr;
                return;
            }

            // Compute per-row byte size from total storage
            const size_t bytes_per_row = total_size / total_rows;
            const size_t src_offset = static_cast<size_t>(row_start) * bytes_per_row;
            const size_t slice_bytes = static_cast<size_t>(N) * bytes_per_row;
            const uint8_t *src = base + src_offset;

            if (weights->is_raw_data_released() || !weights->is_view())
            {
                // Owned or soon-to-be-released data: must copy
                native_blocks_owned_.assign(src, src + slice_bytes);
                native_blocks_ptr_ = native_blocks_owned_.data();
            }
            else
            {
                // mmap view: zero-copy pointer, data survives release_raw_data()
                native_blocks_ptr_ = src;
            }
        }

        // -------------------------------------------------------------------
        // Workspace management for deferred VNNI packing
        // -------------------------------------------------------------------

        /// Shared workspace buffer — reused across all GEMM/GEMV calls.
        /// Thread-safe because inference layers execute sequentially (one GEMM at a time).
        /// The static ensures the workspace survives across calls and avoids repeated
        /// mmap/munmap for each GEMM invocation.
        static AlignedVector<uint8_t> &sharedWorkspace()
        {
            static AlignedVector<uint8_t> ws;
            return ws;
        }

        /// Ensure the workspace is populated with interleaved data from native blocks.
        /// Sets packed_.workspace_data_ to point into the workspace buffer.
        void ensureWorkspace() const
        {
            auto &ws = sharedWorkspace();
            const size_t needed = interleavedWorkspaceSize(packed_);
            if (ws.size() < needed)
                ws.resize_uninitialized(needed);

            repackNativeBlocksToInterleaved(
                native_blocks_ptr_, native_block_size_, packed_, ws.data());

            packed_.setWorkspace(ws.data());
        }

        /// Q8_0 fast path: point workspace directly at native blocks (zero-copy).
        /// The GEMV dispatcher detects Q8_0 and redirects to q8_0_native_gemv()
        /// which reads raw Q8_0Block layout directly.
        void ensureWorkspaceRaw() const
        {
            packed_.setWorkspace(native_blocks_ptr_);
        }

        /// Apply rotation to FP32 activation data, returns pointer to rotated data.
        /// If no rotation is configured, returns the original pointer unchanged.
        const float *maybe_rotate_activation(const float *input, int m, int k) const
        {
            if (!activation_rotation_)
                return input;

            const size_t len = static_cast<size_t>(m) * k;
            if (rotation_scratch_.size() < len)
                rotation_scratch_.resize(len);

            std::memcpy(rotation_scratch_.data(), input, len * sizeof(float));
            activation_rotation_->rotate_rows_inplace(rotation_scratch_.data(), m, k);
            return rotation_scratch_.data();
        }
    };

} // namespace llaminar2::cpu::native_vnni
