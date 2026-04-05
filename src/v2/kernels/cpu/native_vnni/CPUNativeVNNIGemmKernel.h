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
            if (!packWeightsCPUNativeVNNI(weights, packed_, row_start, row_end))
            {
                LOG_ERROR("[CPUNativeVNNIGemmKernel] Failed to pack weights");
                valid_ = false;
                return;
            }
            valid_ = true;

            // For Q8_0 (codebook 19): store native block pointer for M=1 GEMV bypass.
            // This avoids the 3-array packed format (weight+scales+comps) which creates
            // too many memory streams for the hardware prefetcher during decode.
            if (packed_.codebook_id == 19)
            {
                auto *q8 = dynamic_cast<const Q8_0Tensor *>(weights);
                if (q8)
                {
                    int actual_start = (row_start >= 0) ? row_start : 0;
                    native_q8_0_blocks_ = q8->typed_data() +
                                          static_cast<size_t>(actual_start) * packed_.blocks_per_row;
                    native_q8_0_bpr_ = packed_.blocks_per_row;
                }
            }

            LOG_DEBUG("[CPUNativeVNNIGemmKernel] Packed "
                      << packed_.N << "×" << packed_.K
                      << " weights (codebook=" << (int)packed_.codebook_id
                      << ", payload=" << packed_.payload_bytes << " B/block"
                      << ", asymmetric=" << packed_.is_asymmetric
                      << ", native_q8_0=" << (native_q8_0_blocks_ != nullptr) << ")");

            // Pick up activation rotation from the weight tensor (if set).
            // When present, activations will be rotated before Q8_1 quantization
            // to reduce kurtosis and improve int8 fidelity.
            activation_rotation_ = weights->activationRotation();
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
            const MPIContext *mpi_ctx = nullptr,
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

            if (m == 1)
            {
                // Optimized GEMV path
                if (beta == 0.0f && alpha == 1.0f)
                {
                    gemv_native_vnni(packed_, A_data, C_data,
                                     native_q8_0_blocks_, native_q8_0_bpr_);
                }
                else
                {
                    // General case: C = alpha * A@B + beta * C
                    std::vector<float> temp(n);
                    gemv_native_vnni(packed_, A_data, temp.data(),
                                     native_q8_0_blocks_, native_q8_0_bpr_);
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

            // M=1 fast path: call GEMV directly with raw pointer, skip TensorBase wrapper
            if (m == 1 && alpha == 1.0f && beta == 0.0f)
            {
                gemv_native_vnni(packed_, gemm_input, output_fp32,
                                 native_q8_0_blocks_, native_q8_0_bpr_);
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
            const MPIContext *mpi_ctx = nullptr,
            DeviceWorkspaceManager *workspace = nullptr) override
        {
            if (!valid_)
                return false;

            const float *input_data = input->data();

            // Apply activation rotation for kurtosis reduction (if configured)
            input_data = maybe_rotate_activation(input_data, m, k);

            // Check if all projections support native Q8_0 GEMV (M=1 only).
            [[maybe_unused]] bool all_native_q8_0 = false;
#if defined(__AVX512F__)
            if (m == 1)
            {
                all_native_q8_0 = true;
                for (const auto &proj : projections)
                {
                    auto *vnni = dynamic_cast<CPUNativeVNNIGemmKernel *>(proj.kernel);
                    if (!vnni || !vnni->native_q8_0_blocks_)
                    {
                        all_native_q8_0 = false;
                        break;
                    }
                }

                // Fused FP32-dequant path: process all projections in a
                // single OMP region, eliminating fork/join overhead.
                if (all_native_q8_0)
                {
                    const int num_proj = static_cast<int>(projections.size());
                    FusedProjectionDesc descs[8]; // max 8 projections (QKV=3, GateUp=2)
                    for (int p = 0; p < num_proj; ++p)
                    {
                        auto *vnni = static_cast<CPUNativeVNNIGemmKernel *>(projections[p].kernel);
                        descs[p].weights = vnni->native_q8_0_blocks_;
                        descs[p].output = projections[p].output->mutable_data();
                        descs[p].bias = projections[p].bias ? projections[p].bias->data() : nullptr;
                        descs[p].N = projections[p].n;
                        descs[p].bpr = vnni->native_q8_0_bpr_;
                    }

                    q8_0_native_gemv_fused(input_data, descs, num_proj);
                    return true;
                }
            }
#endif

            // Non-Q8_0 fallback: pre-quantize input to Q8_1
            const int K_blocks = (k + 31) / 32;
            std::vector<Q8_1Block> shared_q8;
            if (!all_native_q8_0)
            {
                shared_q8.resize(static_cast<size_t>(m) * K_blocks);
                if (m == 1)
                {
                    const bool k_aligned = (k % 32 == 0);
                    int kb = 0;
#if defined(__AVX512F__)
                    if (k_aligned)
                    {
                        for (; kb + 1 < K_blocks; kb += 2)
                            simd::quantize_two_blocks_avx512(input_data + kb * 32, shared_q8[kb], shared_q8[kb + 1]);
                    }
#endif
                    for (; kb < K_blocks; ++kb)
                    {
                        int block_start = kb * 32;
                        int block_len = std::min(32, k - block_start);
                        simd::quantize_single_block(input_data + block_start, shared_q8[kb], block_len);
                    }
                }
                else
                {
                    quantize_activations_to_q8_1(input_data, shared_q8.data(), m, k, K_blocks);
                }
            }

            // Run each projection + apply epilogues
            for (const auto &proj : projections)
            {
                if (!proj.kernel || !proj.output)
                    return false;

                float *out_data = proj.output->mutable_data();

                auto *vnni_kernel = dynamic_cast<CPUNativeVNNIGemmKernel *>(proj.kernel);
                if (vnni_kernel && vnni_kernel->valid_)
                {
                    const auto &proj_packed = vnni_kernel->packed_;
                    if (m == 1)
                        gemv_native_vnni_preq(proj_packed, shared_q8.data(), out_data);
                    else
                        gemm_native_vnni_preq(proj_packed, shared_q8.data(), out_data, m, proj.n);
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

                // Apply bias epilogue
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

        // Q8_0 native GEMV bypass: pointer into original tensor's contiguous blocks.
        // Set for codebook_id==19 (Q8_0) to enable single-stream GEMV at M=1.
        const Q8_0Block *native_q8_0_blocks_ = nullptr;
        int native_q8_0_bpr_ = 0;

        // Cached scratch buffer for fused SwiGLU+GEMM (avoids malloc per decode token)
        mutable std::vector<float> swiglu_scratch_;

        // Block-diagonal rotation for activation kurtosis reduction.
        // When set, activations are rotated before Q8_1 quantization for GEMM.
        // The weight must have been pre-rotated with the same rotation.
        const ActivationRotation *activation_rotation_ = nullptr;

        // Cached scratch buffer for rotated activation (avoids malloc per call)
        mutable std::vector<float> rotation_scratch_;

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
