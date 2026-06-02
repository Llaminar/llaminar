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
#include "CPUPackedWeights.h"
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

            // Store the native block size for this format. VNNI engines are
            // fully eager: the packed interleaved representation stays owned
            // by the engine and is never rebuilt from raw tensor storage.
            native_block_size_ = native_block_bytes_for_codebook(packed_.codebook_id);

            LOG_DEBUG("[CPUNativeVNNIGemmKernel] Packed "
                      << packed_.N << "×" << packed_.K
                      << " weights (codebook=" << (int)packed_.codebook_id
                      << ", payload=" << packed_.payload_bytes << " B/block"
                      << ", asymmetric=" << packed_.is_asymmetric
                      << ", rotation=" << (activation_rotation_ != nullptr)
                      << ", permanent_interleaved=true)");
        }

        /**
         * @brief Construct from pre-packed weights (move).
         */
        explicit CPUNativeVNNIGemmKernel(CPUNativeVNNIPackedWeights &&packed)
            : packed_(std::move(packed)), valid_(packed_.hasInterleavedData())
        {
            native_block_size_ = native_block_bytes_for_codebook(packed_.codebook_id);
            if (!valid_)
                LOG_ERROR("[CPUNativeVNNIGemmKernel] Pre-packed CPU_NATIVE_VNNI weights are missing eager interleaved data");
        }

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

            // Legacy deferred-packing guard. New CPU VNNI engines are eager and
            // transferred blobs with native-block deferred payloads are rejected.
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
        // Weight Lifecycle (IPackedWeights integration)
        // -------------------------------------------------------------------

        std::unique_ptr<IPackedWeights> detachWeights() override
        {
            if (!valid_)
                return nullptr;

            auto result = std::make_unique<CPUPackedWeights>(std::move(packed_));

            // Invalidate kernel
            packed_ = CPUNativeVNNIPackedWeights{};
            native_blocks_owned_.clear();
            native_blocks_ptr_ = nullptr;
            deferred_packing_ = false;
            valid_ = false;
            return result;
        }

        std::unique_ptr<IPackedWeights> cloneWeights() const override
        {
            if (!valid_)
                return nullptr;

            return std::make_unique<CPUPackedWeights>(packed_);
        }

        bool attachWeights(std::unique_ptr<IPackedWeights> weights) override
        {
            if (!weights || weights->format() != PackedWeightsFormat::CPU_NATIVE_VNNI)
                return false;

            if (dynamic_cast<CPUPackedWeightsWithNativeBlocks *>(weights.get()))
            {
                LOG_ERROR("[CPUNativeVNNIGemmKernel] Deferred/native-block weight attachment is disabled; expected eager interleaved CPU_NATIVE_VNNI weights");
                return false;
            }

            auto *cpu_packed = dynamic_cast<CPUPackedWeights *>(weights.get());
            if (!cpu_packed)
                return false;

            packed_ = cpu_packed->takePacked();
            if (!packed_.hasInterleavedData())
            {
                LOG_ERROR("[CPUNativeVNNIGemmKernel] Attached CPU_NATIVE_VNNI weights have no eager interleaved data");
                packed_ = CPUNativeVNNIPackedWeights{};
                return false;
            }

            native_blocks_owned_.clear();
            native_blocks_ptr_ = nullptr;
            deferred_packing_ = false;

            valid_ = true;
            return true;
        }

        void releaseWeights() override
        {
            {
                CPUNativeVNNIPackedWeights empty;
                packed_ = std::move(empty);
            }
            native_blocks_owned_.clear();
            native_blocks_owned_.shrink_to_fit();
            native_blocks_ptr_ = nullptr;
            deferred_packing_ = false;
            valid_ = false;
        }

        bool hasWeights() const override { return valid_; }

        size_t packedWeightBytes() const override
        {
            if (!valid_)
                return 0;
            return packed_.native_interleaved.size() + packed_.payload.size() + packed_.int8_flat.size() + native_blocks_owned_.size();
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
            float alpha = 1.0f, float beta = 0.0f,
            DeviceWorkspaceManager *workspace = nullptr) override
        {
            (void)workspace;
            if (!valid_)
                return false;
            if (!gate || !up || !output)
            {
                LOG_ERROR("[CPUNativeVNNIGemmKernel] fused SwiGLU received null tensor");
                return false;
            }
            if (m <= 0 || n <= 0 || k <= 0)
            {
                LOG_ERROR("[CPUNativeVNNIGemmKernel] fused SwiGLU invalid dimensions m="
                          << m << " n=" << n << " k=" << k);
                return false;
            }
            if (packed_.N != n || packed_.K != k)
            {
                LOG_ERROR("[CPUNativeVNNIGemmKernel] fused SwiGLU dimension mismatch: packed N="
                          << packed_.N << " K=" << packed_.K << ", call n=" << n
                          << " k=" << k);
                return false;
            }
            const size_t input_size = static_cast<size_t>(m) * static_cast<size_t>(k);
            const size_t output_size = static_cast<size_t>(m) * static_cast<size_t>(n);
            if (gate->numel() < input_size || up->numel() < input_size || output->numel() < output_size)
            {
                LOG_ERROR("[CPUNativeVNNIGemmKernel] fused SwiGLU tensor capacity mismatch: gate="
                          << gate->numel() << " up=" << up->numel() << " output=" << output->numel()
                          << ", required input=" << input_size << " output=" << output_size);
                return false;
            }

            const float *gate_fp32 = gate->data();
            const float *up_fp32 = up->data();
            float *output_fp32 = output->mutable_data();

            // Apply SwiGLU to get the GEMM input: temp = silu(gate) * up  [m, k]

            // Prepared CPU expert engines are shared across graph participants in
            // LocalTP. Keep per-call scratch thread-local so concurrent users do
            // not race on mutable engine state.
            thread_local std::vector<float> swiglu_scratch_tls;
            const size_t needed = input_size;
            if (swiglu_scratch_tls.size() < needed)
                swiglu_scratch_tls.resize(needed);

            // M=1 decode: use serial SwiGLU to avoid OMP fork/join overhead.
            // For MoE experts with intermediate=512, the 512-element SwiGLU
            // takes ~0.1µs in SIMD vs ~6µs OMP barrier cost.
            if (m == 1)
                primitives::compute_swiglu_serial(gate_fp32, up_fp32, swiglu_scratch_tls.data(),
                                                  static_cast<int>(input_size));
            else
                primitives::compute_swiglu(gate_fp32, up_fp32, swiglu_scratch_tls.data(),
                                           static_cast<int>(input_size));

            // Apply activation rotation for kurtosis reduction (if configured)
            const float *gemm_input = maybe_rotate_activation(swiglu_scratch_tls.data(), m, k);

            // Legacy deferred-packing guard; eager engines should never enter it.
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
            // M==1 decode path: try fused single-OMP-region GEMV first.
            // This quantizes the input to Q8_1 once and runs all projections
            // with nowait in a single OMP parallel region, saving:
            //   - (N-1) × Q8_1 quantization (~2μs each)
            //   - (N-1) × OMP fork/join (~6μs each)
            // Falls back to individual calls for non-VNNI kernels.
            // -----------------------------------------------------------
            if (m == 1)
            {
                // Check if ALL projections are CPUNativeVNNIGemmKernel
                bool all_vnni = true;
                for (const auto &proj : projections)
                {
                    if (!proj.kernel || !proj.output)
                        return false;
                    auto *vnni = dynamic_cast<CPUNativeVNNIGemmKernel *>(proj.kernel);
                    if (!vnni || !vnni->valid_)
                    {
                        all_vnni = false;
                        break;
                    }
                }

                if (all_vnni && projections.size() >= 2)
                {
                    // Fused path: quantize once, single OMP region for all projections

                    // Apply activation rotation (if configured on this kernel)
                    input_data = maybe_rotate_activation(input_data, 1, k);

                    // Set up workspace for deferred-packing kernels.
                    // Multiple deferred kernels need simultaneous workspace slots
                    // since the fused GEMV reads all weights in parallel.
                    {
                        size_t max_interleave_ws = 0;
                        int deferred_interleave_count = 0;
                        for (const auto &proj : projections)
                        {
                            auto *vnni = static_cast<CPUNativeVNNIGemmKernel *>(proj.kernel);
                            if (vnni->deferred_packing_ && vnni->packed_.codebook_id != 19)
                            {
                                max_interleave_ws = std::max(max_interleave_ws,
                                                             interleavedWorkspaceSize(vnni->packed_));
                                deferred_interleave_count++;
                            }
                        }
                        if (deferred_interleave_count > 0)
                        {
                            auto &ws = sharedWorkspace();
                            const size_t total = max_interleave_ws * deferred_interleave_count;
                            if (ws.size() < total)
                                ws.resize_uninitialized(total);
                        }
                        int slot_idx = 0;
                        for (const auto &proj : projections)
                        {
                            auto *vnni = static_cast<CPUNativeVNNIGemmKernel *>(proj.kernel);
                            if (vnni->deferred_packing_)
                            {
                                if (vnni->packed_.codebook_id == 19)
                                {
                                    vnni->ensureWorkspaceRaw();
                                }
                                else
                                {
                                    uint8_t *slot = sharedWorkspace().data() +
                                                    static_cast<size_t>(slot_idx) * max_interleave_ws;
                                    repackNativeBlocksToInterleaved(
                                        vnni->native_blocks_ptr_, vnni->native_block_size_,
                                        vnni->packed_, slot);
                                    vnni->packed_.setWorkspace(slot);
                                    slot_idx++;
                                }
                            }
                        }
                    }

                    // Quantize activations to Q8_1 once (shared across all projections)
                    thread_local std::vector<Q8_1Block> fused_q8_tls;
                    if (static_cast<int>(fused_q8_tls.size()) < K_blocks)
                        fused_q8_tls.resize(K_blocks);
                    {
                        int kb = 0;
#if defined(__AVX512F__)
                        for (; kb + 1 < K_blocks; kb += 2)
                            simd::quantize_two_blocks_avx512(input_data + kb * 32,
                                                             fused_q8_tls[kb], fused_q8_tls[kb + 1]);
#endif
                        for (; kb < K_blocks; ++kb)
                            simd::quantize_single_block(input_data + kb * 32, fused_q8_tls[kb],
                                                        std::min(32, k - kb * 32));
                    }

                    // Build fused GEMV descriptors
                    FusedGemvDesc descs[16]; // Stack-allocated, max 16 projections (MoE batches 8 gate+up)
                    int num_descs = 0;
                    for (const auto &proj : projections)
                    {
                        if (num_descs >= 16)
                            break;
                        auto *vnni = static_cast<CPUNativeVNNIGemmKernel *>(proj.kernel);
                        auto &d = descs[num_descs++];
                        d.packed = &vnni->packed_;
                        d.output = proj.output->mutable_data();
                        d.bias = proj.bias ? proj.bias->data() : nullptr;
                        d.N = proj.n;
                        d.bpr = K_blocks;

                        // Check for Q8_0 raw path (deferred zero-copy)
                        if (vnni->packed_.codebook_id == 19 && vnni->packed_.workspace_data_)
                            d.q8_0_raw = reinterpret_cast<const Q8_0Block *>(vnni->packed_.workspace_data_);
                        else
                            d.q8_0_raw = nullptr;
                    }

                    // Single OMP region with nowait between projections
                    gemv_native_vnni_fused_preq(fused_q8_tls.data(), descs, num_descs);

                    // Clean up deferred workspace
                    for (const auto &proj : projections)
                    {
                        auto *vnni = static_cast<CPUNativeVNNIGemmKernel *>(proj.kernel);
                        if (vnni->deferred_packing_)
                            vnni->packed_.clearWorkspace();
                    }

                    return true;
                }

                // Fallback: individual calls for non-VNNI or single projection
                for (const auto &proj : projections)
                {
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

        // =====================================================================
        // Fused multi-input GEMV for MoE expert down projections
        // =====================================================================

        bool multiply_fused_expert_down(
            const FusedExpertDownDesc *descs, int num_descs,
            int m, int k) override
        {
            if (m != 1 || num_descs < 1)
                return false;

            // Verify all kernels are CPUNativeVNNIGemmKernel
            for (int i = 0; i < num_descs; ++i)
            {
                auto *vnni = dynamic_cast<CPUNativeVNNIGemmKernel *>(descs[i].kernel);
                if (!vnni || !vnni->valid_)
                    return false;
            }

            const int K_blocks = (k + 31) / 32;

            // Set up deferred workspace for all kernels
            {
                size_t max_interleave_ws = 0;
                int deferred_interleave_count = 0;
                for (int i = 0; i < num_descs; ++i)
                {
                    auto *vnni = static_cast<CPUNativeVNNIGemmKernel *>(descs[i].kernel);
                    if (vnni->deferred_packing_ && vnni->packed_.codebook_id != 19)
                    {
                        max_interleave_ws = std::max(max_interleave_ws,
                                                     interleavedWorkspaceSize(vnni->packed_));
                        deferred_interleave_count++;
                    }
                }
                if (deferred_interleave_count > 0)
                {
                    auto &ws = sharedWorkspace();
                    const size_t total = max_interleave_ws * deferred_interleave_count;
                    if (ws.size() < total)
                        ws.resize_uninitialized(total);
                }
                int slot_idx = 0;
                for (int i = 0; i < num_descs; ++i)
                {
                    auto *vnni = static_cast<CPUNativeVNNIGemmKernel *>(descs[i].kernel);
                    if (vnni->deferred_packing_)
                    {
                        if (vnni->packed_.codebook_id == 19)
                        {
                            vnni->ensureWorkspaceRaw();
                        }
                        else
                        {
                            uint8_t *slot = sharedWorkspace().data() +
                                            static_cast<size_t>(slot_idx) * max_interleave_ws;
                            repackNativeBlocksToInterleaved(
                                vnni->native_blocks_ptr_, vnni->native_block_size_,
                                vnni->packed_, slot);
                            vnni->packed_.setWorkspace(slot);
                            slot_idx++;
                        }
                    }
                }
            }

            // Quantize each expert's FP32 input to Q8_1
            // Use a contiguous buffer for all experts' Q8_1 blocks
            thread_local std::vector<Q8_1Block> multi_q8_tls;
            const size_t total_blocks = static_cast<size_t>(num_descs) * K_blocks;
            if (multi_q8_tls.size() < total_blocks)
                multi_q8_tls.resize(total_blocks);

            for (int i = 0; i < num_descs; ++i)
            {
                Q8_1Block *A_q8 = multi_q8_tls.data() + static_cast<size_t>(i) * K_blocks;
                const float *input_data = descs[i].input;
                int kb = 0;
#if defined(__AVX512F__)
                for (; kb + 1 < K_blocks; kb += 2)
                    simd::quantize_two_blocks_avx512(input_data + kb * 32,
                                                     A_q8[kb], A_q8[kb + 1]);
#endif
                for (; kb < K_blocks; ++kb)
                    simd::quantize_single_block(input_data + kb * 32, A_q8[kb],
                                                std::min(32, k - kb * 32));
            }

            // Build fused multi-input GEMV descriptors
            FusedGemvMultiInputDesc mi_descs[16]; // max 16 experts
            int num_mi = 0;
            for (int i = 0; i < num_descs && num_mi < 16; ++i)
            {
                auto *vnni = static_cast<CPUNativeVNNIGemmKernel *>(descs[i].kernel);
                auto &d = mi_descs[num_mi++];
                d.A_q8 = multi_q8_tls.data() + static_cast<size_t>(i) * K_blocks;
                d.packed = &vnni->packed_;
                d.output = descs[i].output;
                d.N = descs[i].n;
                d.bpr = K_blocks;
                d.q8_0_raw = (vnni->packed_.codebook_id == 19 && vnni->packed_.workspace_data_)
                                 ? reinterpret_cast<const Q8_0Block *>(vnni->packed_.workspace_data_)
                                 : nullptr;
            }

            // Single OMP region with nowait between expert projections
            gemv_fused_multi_input_preq(mi_descs, num_mi);

            // Clean up deferred workspace
            for (int i = 0; i < num_descs; ++i)
            {
                auto *vnni = static_cast<CPUNativeVNNIGemmKernel *>(descs[i].kernel);
                if (vnni->deferred_packing_)
                    vnni->packed_.clearWorkspace();
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
        /// Legacy deferred-packing storage. Kept only for defensive cleanup paths;
        /// new CPU VNNI engines keep eager interleaved packed weights.
        std::vector<uint8_t> native_blocks_owned_;

        /// Pointer to native block data — either into native_blocks_owned_
        /// (TP-sliced weights) or into the original mmap region (non-TP views).
        const uint8_t *native_blocks_ptr_ = nullptr;

        /// Size in bytes of a single native quantized block (34 for Q8_0, 18 for Q4_0, etc.)
        size_t native_block_size_ = 0;

        /// Legacy flag. New CPU VNNI engine construction never sets this true.
        bool deferred_packing_ = false;

        // Cached Q8_1 quantization buffer for fused projections (avoids malloc per decode token)
        mutable std::vector<Q8_1Block> q8_scratch_;

        // Block-diagonal rotation for activation kurtosis reduction.
        // When set, activations are rotated before Q8_1 quantization for GEMM.
        // The weight must have been pre-rotated with the same rotation.
        const ActivationRotation *activation_rotation_ = nullptr;

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
        // Q4_K superblock → Q4_1 elementary block synthesis
        // -------------------------------------------------------------------

        /// Synthesize Q4_1-compatible elementary blocks from Q4_K superblock data.
        ///
        /// Each Q4_K superblock (144 bytes, 256 elements) is decomposed into
        /// 8 elementary blocks (20 bytes each, 32 elements). The elementary
        /// block format matches Q4_1: [scale_fp16(2) | min_fp16(2) | payload(16)].
        ///
        /// The resulting blocks are stored in native_blocks_owned_ and can be
        /// repacked by the standard repackNativeBlocksToInterleaved() function
        /// since packed_.codebook_id == 5 (Q4_1).
        ///
        /// Legacy deferred-packing helper retained for experiments only; normal
        /// CPU VNNI engine construction does not call it.
        void synthesizeElementaryBlocksFromSuperblock(const TensorBase *weights,
                                                      int row_start, int row_end)
        {
            if (row_start < 0)
                row_start = 0;
            if (row_end < 0)
                row_end = weights->shape()[0];

            const int N = row_end - row_start;
            const int K = packed_.K;
            const int bpr = packed_.blocks_per_row; // elementary blocks per row (K/32)
            const int sbpr = K / 256;               // superblocks per row

            // Elementary Q4_1 block: scale_fp16(2) + min_fp16(2) + payload(16) = 20 bytes
            static constexpr size_t ELEM_BLOCK_SIZE = 20;

            const auto *base = reinterpret_cast<const uint8_t *>(weights->raw_data());

            if (!base || sbpr == 0)
            {
                LOG_WARN("[CPUNativeVNNIGemmKernel] Cannot synthesize elementary blocks: "
                         << "raw_data()=" << (const void *)base
                         << " sbpr=" << sbpr
                         << " — keeping permanent interleaved data");
                native_blocks_ptr_ = nullptr;
                return;
            }

            // Q4_K superblock layout: row_stride = sbpr * sizeof(Q4_KBlock)
            // Compute from shape instead of size_bytes() (views may report 0).
            static constexpr size_t Q4K_BLOCK_SIZE = 144; // sizeof(Q4_KBlock)
            const size_t sb_row_stride = static_cast<size_t>(sbpr) * Q4K_BLOCK_SIZE;
            const uint8_t *row_base = base + static_cast<size_t>(row_start) * sb_row_stride;

            // Allocate elementary blocks: N rows × bpr blocks × 20 bytes each
            native_blocks_owned_.resize(static_cast<size_t>(N) * bpr * ELEM_BLOCK_SIZE);

#pragma omp parallel for schedule(static)
            for (int n = 0; n < N; ++n)
            {
                const uint8_t *row_ptr = row_base + static_cast<size_t>(n) * sb_row_stride;

                for (int sb = 0; sb < sbpr; ++sb)
                {
                    const auto *blk = reinterpret_cast<const Q4_KBlock *>(
                        row_ptr + static_cast<size_t>(sb) * Q4K_BLOCK_SIZE);

                    const float d = fp16_to_fp32(blk->d);
                    const float dmin = fp16_to_fp32(blk->dmin);

                    for (int sub = 0; sub < 8; ++sub)
                    {
                        const int kb = sb * 8 + sub;
                        uint8_t *dst = native_blocks_owned_.data() +
                                       (static_cast<size_t>(n) * bpr + kb) * ELEM_BLOCK_SIZE;

                        // Decode 6-bit packed scale and min for this sub-block
                        uint8_t sc, m_val;
                        simd::get_scale_min_k4(sub, blk->scales, &sc, &m_val);

                        // Convert to FP16 (matches packVnniBlock() output)
                        const uint16_t scale_fp16 = fp32_to_fp16(d * static_cast<float>(sc));
                        const uint16_t min_fp16 = fp32_to_fp16(-dmin * static_cast<float>(m_val));

                        std::memcpy(dst, &scale_fp16, 2);
                        std::memcpy(dst + 2, &min_fp16, 2);

                        // Extract nibble payload: repack from Q4_K interleaved layout
                        // to contiguous 16 bytes (matches packVnniBlock() nibble extraction)
                        const int group_idx = sub / 2;
                        const int is_high = sub & 1;
                        const uint8_t *src32 = blk->qs + group_idx * 32;

                        if (is_high)
                        {
                            for (int i = 0; i < 16; ++i)
                                dst[4 + i] = (src32[i] >> 4) | (src32[i + 16] & 0xF0);
                        }
                        else
                        {
                            for (int i = 0; i < 16; ++i)
                                dst[4 + i] = (src32[i] & 0xF) | ((src32[i + 16] & 0xF) << 4);
                        }
                    }
                }
            }

            native_blocks_ptr_ = native_blocks_owned_.data();
            // Override native_block_size_ to elementary Q4_1 block size
            // (the codebook_id-based default of 20 already matches, but be explicit)
            native_block_size_ = ELEM_BLOCK_SIZE;
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
            thread_local std::vector<float> rotation_scratch_tls;
            if (rotation_scratch_tls.size() < len)
                rotation_scratch_tls.resize(len);

            std::memcpy(rotation_scratch_tls.data(), input, len * sizeof(float));
            activation_rotation_->rotate_rows_inplace(rotation_scratch_tls.data(), m, k);
            return rotation_scratch_tls.data();
        }
    };

} // namespace llaminar2::cpu::native_vnni
