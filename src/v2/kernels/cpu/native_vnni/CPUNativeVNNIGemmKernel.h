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
            LOG_DEBUG("[CPUNativeVNNIGemmKernel] Packed "
                      << packed_.N << "×" << packed_.K
                      << " weights (codebook=" << (int)packed_.codebook_id
                      << ", payload=" << packed_.payload_bytes << " B/block"
                      << ", asymmetric=" << packed_.is_asymmetric << ")");
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
         * For M=1: dispatches to optimized GEMV path.
         * For M>1: dispatches to row-by-row GEMV (phase 1; tiled GEMM is phase 2).
         */
        bool multiply(
            const float *A, float *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            DeviceWorkspaceManager *workspace = nullptr) override
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

            // Handle beta scaling of existing C
            if (beta != 0.0f && beta != 1.0f)
            {
                for (int i = 0; i < m * n; ++i)
                    C[i] *= beta;
            }
            else if (beta == 0.0f)
            {
                // GEMM will overwrite C completely
            }

            if (m == 1)
            {
                // Optimized GEMV path
                if (beta == 0.0f && alpha == 1.0f)
                {
                    gemv_native_vnni(packed_, A, C);
                }
                else
                {
                    // General case: C = alpha * A@B + beta * C
                    // Use temp buffer for raw GEMV result, then blend
                    std::vector<float> temp(n);
                    gemv_native_vnni(packed_, A, temp.data());
                    for (int j = 0; j < n; ++j)
                        C[j] += alpha * temp[j];
                }
            }
            else
            {
                // M>1: row-by-row GEMV for now
                if (beta == 0.0f && alpha == 1.0f)
                {
                    gemm_native_vnni(packed_, A, C, m, n);
                }
                else
                {
                    std::vector<float> temp(n);
                    for (int row = 0; row < m; ++row)
                    {
                        gemv_native_vnni(packed_, A + row * k, temp.data());
                        for (int j = 0; j < n; ++j)
                            C[row * n + j] += alpha * temp[j];
                    }
                }
            }

            return true;
        }

        /**
         * @brief Tensor-aware multiply (delegates to raw multiply)
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
            const float *A_data = A->data() + activation_row_offset * k;
            float *C_data = C->mutable_data();

            bool success = multiply(A_data, C_data, m, n, k, transpose_B, alpha, beta, mpi_ctx, device_idx, workspace);
            if (!success)
                return false;

            // Apply bias epilogue: C[m, j] += bias[j]
            if (bias)
            {
                const float *bias_data = bias->data();
                apply_bias_epilogue(C_data, bias_data, m, n, n);
            }

            return true;
        }

        /**
         * @brief Activation-activation multiply (not applicable for weight GEMM).
         */
        bool multiply_activations(
            const float *A, const float *B, float *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            (void)A;
            (void)B;
            (void)C;
            (void)m;
            (void)n;
            (void)k;
            (void)transpose_B;
            (void)alpha;
            (void)beta;
            (void)mpi_ctx;
            (void)device_idx;
            return false; // Not applicable for weight GEMM
        }

        /**
         * @brief Strided activation-activation multiply (not applicable for weight GEMM).
         */
        bool multiply_activations_strided(
            const float *A, const float *B, float *C,
            int m, int n, int k,
            int lda, int ldb, int ldc,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
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
            return false; // Not applicable for weight GEMM
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
            std::vector<float> swiglu_input(input_size);
            primitives::compute_swiglu(gate_fp32, up_fp32, swiglu_input.data(),
                                       static_cast<int>(input_size));

            // GEMM: output = W_down @ swiglu_input
            return multiply(swiglu_input.data(), output_fp32, m, n, k,
                            true, alpha, beta, nullptr, -1, nullptr);
        }

        // -------------------------------------------------------------------
        // Fused multi-projection with quantize-once + epilogues
        // -------------------------------------------------------------------

        bool supports_fused_projection() const override
        {
            return true;
        }

        bool multiply_fused(
            const float *input,
            const std::vector<FusedProjectionDesc> &projections,
            int m, int k,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            DeviceWorkspaceManager *workspace = nullptr) override
        {
            (void)mpi_ctx;
            (void)workspace;

            if (!valid_ || device_idx != -1)
                return false;

            // Pre-quantize input once (shared across all projections)
            const int K_blocks = (k + 31) / 32;
            std::vector<Q8_1Block> shared_q8(static_cast<size_t>(m) * K_blocks);

            if (m == 1)
            {
                // Single-row: quantize on current thread
                const bool k_aligned = (k % 32 == 0);
                int kb = 0;
#if defined(__AVX512F__)
                if (k_aligned)
                {
                    for (; kb + 1 < K_blocks; kb += 2)
                        simd::quantize_two_blocks_avx512(input + kb * 32, shared_q8[kb], shared_q8[kb + 1]);
                }
#endif
                for (; kb < K_blocks; ++kb)
                {
                    int block_start = kb * 32;
                    int block_len = std::min(32, k - block_start);
                    simd::quantize_single_block(input + block_start, shared_q8[kb], block_len);
                }
            }
            else
            {
                // Multi-row: parallel quantization
                quantize_activations_to_q8_1(input, shared_q8.data(), m, k, K_blocks);
            }

            // Run each projection using the shared Q8_1 buffer + apply epilogues
            for (const auto &proj : projections)
            {
                if (!proj.kernel)
                    return false;

                // Check if this projection uses OUR packed weights
                auto *vnni_kernel = dynamic_cast<CPUNativeVNNIGemmKernel *>(proj.kernel);
                if (vnni_kernel && vnni_kernel->valid_)
                {
                    // Fast path: use pre-quantized GEMV/GEMM directly
                    const auto &proj_packed = vnni_kernel->packed_;
                    if (m == 1)
                        gemv_native_vnni_preq(proj_packed, shared_q8.data(), proj.output);
                    else
                        gemm_native_vnni_preq(proj_packed, shared_q8.data(), proj.output, m, proj.n);
                }
                else
                {
                    // Fallback: use generic multiply (re-quantizes internally)
                    bool success = proj.kernel->multiply(
                        input, proj.output, m, proj.n, k,
                        true, 1.0f, 0.0f, mpi_ctx, device_idx, workspace);
                    if (!success)
                        return false;
                }

                // Apply bias epilogue
                if (proj.bias)
                {
                    const float *bias_data = proj.bias->data();
                    apply_bias_epilogue(proj.output, bias_data, m, proj.n, proj.n);
                }
            }
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

            // Pre-quantize input once (shared across all projections)
            const int K_blocks = (k + 31) / 32;
            std::vector<Q8_1Block> shared_q8(static_cast<size_t>(m) * K_blocks);

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

            // Run each projection using the shared Q8_1 buffer + apply epilogues
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
                    // multiply_tensor already handled bias for fallback
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
    };

} // namespace llaminar2::cpu::native_vnni
