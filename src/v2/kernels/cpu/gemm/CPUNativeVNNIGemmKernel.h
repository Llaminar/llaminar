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
#include "utils/PerfStatsCollector.h"

#include <atomic>

namespace llaminar2::cpu::native_vnni
{

    /**
     * @brief RAII publication of serial TP output geometry for CPU NativeVNNI.
     *
     * The scope changes dispatch geometry only on the caller thread. The full
     * replicated weight, complete output buffer, and physical N traversal stay
     * unchanged. Nested graph/stage scopes restore their predecessor exactly,
     * which keeps concurrent rank workers independent without locks.
     */
    class CPUNativeVNNIOutputPartitionEquivalenceScope final
        : public ITensorGemm::OutputPartitionEquivalenceScope
    {
    public:
        /** Publish @p serial_partition_n until this object is destroyed. */
        explicit CPUNativeVNNIOutputPartitionEquivalenceScope(
            int serial_partition_n)
            : previous_(cpuNativeVNNISerialOutputPartitionN())
        {
            setCPUNativeVNNISerialOutputPartitionN(serial_partition_n);
        }

        /** Restore the enclosing output-partition policy. */
        ~CPUNativeVNNIOutputPartitionEquivalenceScope() override
        {
            setCPUNativeVNNISerialOutputPartitionN(previous_);
        }

        CPUNativeVNNIOutputPartitionEquivalenceScope(
            const CPUNativeVNNIOutputPartitionEquivalenceScope &) = delete;
        CPUNativeVNNIOutputPartitionEquivalenceScope &operator=(
            const CPUNativeVNNIOutputPartitionEquivalenceScope &) = delete;

    private:
        int previous_ = 0; ///< Scope value restored at transaction completion.
    };

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
         * @param numerical_policy Arithmetic contract carried by the prepared engine.
         * @param placement Final CPU storage placement established before packing.
         */
        explicit CPUNativeVNNIGemmKernel(const TensorBase *weights,
                                         int row_start = 0, int row_end = -1,
                                         CPUProjectionNumericalPolicy numerical_policy =
                                             CPUProjectionNumericalPolicy::BackendNative,
                                         CPUWeightStoragePlacement placement = CPUWeightStoragePlacement::local())
            : numerical_policy_(numerical_policy)
        {
            // Pick up activation rotation from the weight tensor (if set).
            // When present, activations will be rotated before Q8_1 quantization
            // to reduce kurtosis and improve int8 fidelity.
            // The rotation is also fused into weight packing (dequant→rotate→requant)
            // so that the original tensor format is preserved.
            activation_rotation_ = weights->activationRotation();

            if (!packWeightsCPUNativeVNNI(weights, packed_, row_start, row_end,
                                          activation_rotation_, placement))
            {
                LOG_ERROR("[CPUNativeVNNIGemmKernel] Failed to pack weights");
                valid_ = false;
                return;
            }
            packed_.numerical_policy = numerical_policy_;
            valid_ = true;

            LOG_TRACE("[CPUNativeVNNIGemmKernel] Packed "
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
        explicit CPUNativeVNNIGemmKernel(
            CPUNativeVNNIPackedWeights &&packed,
            CPUProjectionNumericalPolicy numerical_policy =
                CPUProjectionNumericalPolicy::BackendNative)
            : packed_(std::move(packed)),
              valid_(packed_.hasInterleavedData()),
              numerical_policy_(numerical_policy)
        {
            packed_.numerical_policy = numerical_policy_;
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

        /** @brief Export exact CPU-packed source codebook and superblock identity. */
        bool exportNativeVNNISourceIdentity(
            NativeVnniSourceIdentity &out) const override
        {
            if (!valid_)
            {
                out = {};
                return false;
            }
            out = {
                .codebook_id = packed_.codebook_id,
                .is_superblock = packed_.is_superblock,
                .present = true,
            };
            return native_vnni_formats::forSourceIdentity(
                       out.codebook_id, out.is_superblock) != nullptr;
        }

        // -------------------------------------------------------------------
        // ITensorGemm interface
        // -------------------------------------------------------------------

        /**
         * @brief Bind mirrored full-width execution to serial TP shard geometry.
         *
         * CPU NativeVNNI output columns are independent once activation Q8_1
         * quantization is complete. Reusing the serial shard's generated policy,
         * N-task width, and K-partition tree therefore produces exactly the same
         * bytes while one replicated invocation writes all vocabulary columns.
         *
         * @param actual_output_columns Full output width owned by this kernel.
         * @param serial_partition_columns Column width used by serial TP decode.
         * @return A caller-thread RAII scope restored after the LM-head launch.
         */
        std::unique_ptr<OutputPartitionEquivalenceScope>
        beginOutputPartitionEquivalenceScope(
            int actual_output_columns,
            int serial_partition_columns) override
        {
            if (actual_output_columns <= 0 ||
                serial_partition_columns <= 0 ||
                actual_output_columns != packed_.N ||
                serial_partition_columns > actual_output_columns)
            {
                throw std::invalid_argument(
                    "[CPUNativeVNNIGemmKernel] Invalid replicated-output serial partition contract");
            }
            return std::make_unique<
                CPUNativeVNNIOutputPartitionEquivalenceScope>(
                serial_partition_columns);
        }

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

            if (n != packed_.N || k != packed_.K)
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

            multiply_native_vnni_with_epilogue(
                packed_, A_data, C_data, m, n, alpha, beta);

            // Apply bias epilogue: C[m, j] += bias[j]
            if (bias)
            {
                const float *bias_data = bias->data();
                apply_bias_epilogue(C_data, bias_data, m, n, n);
            }

            return true;
        }

        /** @brief Borrow the engine-owned final CPU migration representation. */
        const CPUNativeVNNIPackedWeights *
        exportCPUNativeVNNIPackedWeights() const override
        {
            return valid_ ? &packed_ : nullptr;
        }

        /**
         * @brief Expose final storage to the ticket-gated overlay slot arena.
         *
         * The arena holds this engine for the model lifetime and writes only
         * while its physical slot is inactive. Returning the vector's existing
         * range does not resize, allocate, or change the GEMM object's address.
         */
        std::span<std::uint8_t>
        exportRetiredCPUNativeVNNIStorage() noexcept override
        {
            if (!valid_ || packed_.native_interleaved.empty())
                return {};
            return {
                packed_.native_interleaved.data(),
                packed_.native_interleaved.size(),
            };
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

            auto *cpu_packed = dynamic_cast<CPUPackedWeights *>(weights.get());
            if (!cpu_packed)
                return false;

            // Portable cross-backend records may carry an additional native
            // block section.  CPU attachment consumes only the eager
            // interleaved representation and lets the wrapper release the
            // unneeded cross-backend section after this move.
            packed_ = cpu_packed->takePacked();
            packed_.numerical_policy = numerical_policy_;
            if (!packed_.hasInterleavedData())
            {
                LOG_ERROR("[CPUNativeVNNIGemmKernel] Attached CPU_NATIVE_VNNI weights have no eager interleaved data");
                packed_ = CPUNativeVNNIPackedWeights{};
                return false;
            }

            valid_ = true;
            return true;
        }

        void releaseWeights() override
        {
            {
                CPUNativeVNNIPackedWeights empty;
                packed_ = std::move(empty);
            }
            valid_ = false;
        }

        bool hasWeights() const override { return valid_; }

        size_t packedWeightBytes() const override
        {
            if (!valid_)
                return 0;
            return packed_.native_interleaved.size() +
                   packed_.payload.size() +
                   packed_.int8_flat.size();
        }

        bool canReleaseSourceWeightTensor() const override
        {
            return valid_ && !packed_.native_interleaved.empty();
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
            thread_local AlignedVector<float> swiglu_scratch_tls;
            const size_t needed = input_size;
            if (swiglu_scratch_tls.size() < needed)
                swiglu_scratch_tls.resize_uninitialized(needed);

            // M=1 decode: use serial SwiGLU to avoid OMP fork/join overhead.
            // For MoE experts with intermediate=512, the 512-element SwiGLU
            // takes ~0.1µs in SIMD vs ~6µs OMP barrier cost.
            if (numerical_policy_ ==
                CPUProjectionNumericalPolicy::GPUAlignedExpert)
            {
                if (m == 1)
                {
                    primitives::compute_swiglu_gpu_aligned_expert_serial(
                        gate_fp32,
                        up_fp32,
                        swiglu_scratch_tls.data(),
                        static_cast<int>(input_size));
                }
                else
                {
                    primitives::compute_swiglu_gpu_aligned_expert(
                        gate_fp32,
                        up_fp32,
                        swiglu_scratch_tls.data(),
                        static_cast<int>(input_size));
                }
            }
            else if (m == 1)
            {
                primitives::compute_swiglu_serial(
                    gate_fp32,
                    up_fp32,
                    swiglu_scratch_tls.data(),
                    static_cast<int>(input_size));
            }
            else
            {
                primitives::compute_swiglu(
                    gate_fp32,
                    up_fp32,
                    swiglu_scratch_tls.data(),
                    static_cast<int>(input_size));
            }

            // Apply activation rotation for kurtosis reduction (if configured)
            const float *gemm_input = maybe_rotate_activation(swiglu_scratch_tls.data(), m, k);

            multiply_native_vnni_with_epilogue(
                packed_, gemm_input, output_fp32, m, n, alpha, beta);
            return true;
        }

        /**
         * @brief Grouped MTP verifier SwiGLU + down projection.
         *
         * This follows the same SwiGLU math as the ordinary CPU down path, but
         * sends the resulting runtime-M activation rows through
         * gemm_native_vnni_preq_decode_equivalent_rows().  That helper shares
         * Q8_1 activation quantization across rows and parallelizes the work,
         * while each row keeps the same K-tile reduction order as M=1 decode.
         */
        bool multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
            const TensorBase *gate,
            const TensorBase *up,
            TensorBase *output,
            int m, int n, int k,
            float alpha = 1.0f, float beta = 0.0f,
            DeviceWorkspaceManager *workspace = nullptr) override
        {
            (void)workspace;
            if (!valid_ || !gate || !up || !output || m <= 1 || n <= 0 || k <= 0)
            {
                LOG_ERROR("[CPUNativeVNNIGemmKernel] grouped verifier SwiGLU rejected: valid="
                          << valid_ << " gate=" << (gate != nullptr)
                          << " up=" << (up != nullptr)
                          << " output=" << (output != nullptr)
                          << " m=" << m << " n=" << n << " k=" << k);
                return false;
            }
            if (alpha != 1.0f || beta != 0.0f)
            {
                LOG_ERROR("[CPUNativeVNNIGemmKernel] grouped verifier SwiGLU only supports alpha=1,beta=0; got alpha="
                          << alpha << " beta=" << beta);
                return false;
            }
            if (packed_.N != n || packed_.K != k)
            {
                LOG_ERROR("[CPUNativeVNNIGemmKernel] grouped verifier SwiGLU dimension mismatch: packed N="
                          << packed_.N << " K=" << packed_.K << ", call n=" << n << " k=" << k);
                return false;
            }

            const size_t input_size = static_cast<size_t>(m) * static_cast<size_t>(k);
            const size_t output_size = static_cast<size_t>(m) * static_cast<size_t>(n);
            if (gate->numel() < input_size || up->numel() < input_size || output->numel() < output_size)
            {
                LOG_ERROR("[CPUNativeVNNIGemmKernel] grouped verifier SwiGLU tensor capacity mismatch");
                return false;
            }

            const float *gate_fp32 = gate->data();
            const float *up_fp32 = up->data();
            float *output_fp32 = output->mutable_data();
            if (!gate_fp32 || !up_fp32 || !output_fp32)
            {
                LOG_ERROR("[CPUNativeVNNIGemmKernel] grouped verifier SwiGLU requires FP32 host tensors");
                return false;
            }

            thread_local AlignedVector<float> swiglu_scratch_tls;
            if (swiglu_scratch_tls.size() < input_size)
                swiglu_scratch_tls.resize_uninitialized(input_size);
            const bool perf_enabled =
                PerfStatsCollector::isDomainEnabled("kernel");
            auto perf_start = perf_enabled ? PerfStatsCollector::Clock::now()
                                           : PerfStatsCollector::Clock::time_point{};
            if (numerical_policy_ ==
                CPUProjectionNumericalPolicy::GPUAlignedExpert)
            {
                primitives::compute_swiglu_gpu_aligned_expert(
                    gate_fp32,
                    up_fp32,
                    swiglu_scratch_tls.data(),
                    static_cast<int>(input_size));
            }
            else
            {
                primitives::compute_swiglu(
                    gate_fp32,
                    up_fp32,
                    swiglu_scratch_tls.data(),
                    static_cast<int>(input_size));
            }
            recordVerifierTiming(
                "cpu_native_vnni_verifier_swiglu_compute",
                perf_start,
                m,
                n,
                k,
                1);

            const float *gemm_input = maybe_rotate_activation(swiglu_scratch_tls.data(), m, k);
            const int K_blocks = (k + 31) / 32;
            const size_t shared_q8_blocks = static_cast<size_t>(m) * K_blocks;
            thread_local AlignedVector<Q8_1Block> shared_q8_tls;
            if (shared_q8_tls.size() < shared_q8_blocks)
                shared_q8_tls.resize_uninitialized(shared_q8_blocks);
            perf_start = perf_enabled ? PerfStatsCollector::Clock::now()
                                      : PerfStatsCollector::Clock::time_point{};
            quantize_activations_to_q8_1(
                gemm_input,
                shared_q8_tls.data(),
                m,
                k,
                K_blocks,
                numerical_policy_);
            recordVerifierTiming(
                "cpu_native_vnni_verifier_activation_quantize",
                perf_start,
                m,
                n,
                k,
                1);

            perf_start = perf_enabled ? PerfStatsCollector::Clock::now()
                                      : PerfStatsCollector::Clock::time_point{};
            gemm_native_vnni_preq_decode_equivalent_rows(
                packed_,
                shared_q8_tls.data(),
                output_fp32,
                m,
                n);
            recordVerifierTiming(
                "cpu_native_vnni_verifier_swiglu_down_gemv",
                perf_start,
                m,
                n,
                k,
                1);
            if (PerfStatsCollector::isDomainEnabled("kernel"))
            {
                PerfStatsCollector::addCounter(
                    "kernel",
                    "cpu_native_vnni_grouped_verifier_swiglu_down_calls",
                    1.0,
                    "gemm",
                    "cpu",
                    PerfStatsCollector::Tags{
                        {"m", std::to_string(m)},
                        {"n", std::to_string(n)},
                        {"k", std::to_string(k)}});
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
            const IMPIContext *mpi_ctx = nullptr,
            DeviceWorkspaceManager *workspace = nullptr) override
        {
            (void)mpi_ctx;
            (void)workspace;
            constexpr size_t kMaxFusedProjections = 16u;
            if (!valid_ || !input || m <= 0 || k <= 0 ||
                k != packed_.K || projections.empty() ||
                projections.size() > kMaxFusedProjections)
            {
                LOG_ERROR("[CPUNativeVNNIGemmKernel] fused projection rejected: valid="
                          << valid_ << " input=" << (input != nullptr)
                          << " m=" << m << " k=" << k
                          << " packed_k=" << packed_.K
                          << " projections=" << projections.size());
                return false;
            }

            const float *input_data = input->data();
            if (!input_data || input->numel() < static_cast<size_t>(m) * k)
                return false;

            std::array<CPUNativeVNNIGemmKernel *, kMaxFusedProjections>
                vnni_kernels = {};
            std::array<float *, kMaxFusedProjections> outputs = {};
            std::array<const float *, kMaxFusedProjections> biases = {};
            for (size_t i = 0; i < projections.size(); ++i)
            {
                const auto &projection = projections[i];
                auto *vnni = dynamic_cast<CPUNativeVNNIGemmKernel *>(
                    projection.kernel);
                if (!vnni || !vnni->valid_ || !projection.output ||
                    projection.n <= 0 || vnni->packed_.K != k ||
                    vnni->packed_.N != projection.n ||
                    vnni->activation_rotation_ != activation_rotation_ ||
                    projection.output->numel() <
                        static_cast<size_t>(m) * projection.n ||
                    (projection.bias &&
                     projection.bias->numel() < static_cast<size_t>(projection.n)))
                {
                    LOG_ERROR("[CPUNativeVNNIGemmKernel] fused projection contract "
                              "mismatch at index " << i);
                    return false;
                }

                vnni_kernels[i] = vnni;
                outputs[i] = projection.output->mutable_data();
                biases[i] = projection.bias ? projection.bias->data() : nullptr;
                if (!outputs[i] || (projection.bias && !biases[i]))
                    return false;
            }

            input_data = maybe_rotate_activation(input_data, m, k);
            const int K_blocks = (k + Q8_1Block::BLOCK_SIZE - 1) /
                                 Q8_1Block::BLOCK_SIZE;
            const size_t required_blocks =
                static_cast<size_t>(m) * static_cast<size_t>(K_blocks);
            thread_local AlignedVector<Q8_1Block> shared_q8_tls;
            if (shared_q8_tls.size() < required_blocks)
                shared_q8_tls.resize_uninitialized(required_blocks);
            quantize_activations_to_q8_1(
                input_data,
                shared_q8_tls.data(),
                m,
                k,
                K_blocks,
                numerical_policy_);

            if (m == 1)
            {
                std::array<FusedGemvDesc, kMaxFusedProjections> descriptors = {};
                for (size_t i = 0; i < projections.size(); ++i)
                {
                    descriptors[i] = {
                        .packed = &vnni_kernels[i]->packed_,
                        .output = outputs[i],
                        .bias = biases[i],
                        .N = projections[i].n,
                    };
                }
                gemv_native_vnni_fused_preq(
                    shared_q8_tls.data(),
                    descriptors.data(),
                    static_cast<int>(projections.size()));
                return true;
            }

            for (size_t i = 0; i < projections.size(); ++i)
            {
                gemm_native_vnni_preq(
                    vnni_kernels[i]->packed_,
                    shared_q8_tls.data(),
                    outputs[i],
                    m,
                    projections[i].n);
                if (biases[i])
                {
                    apply_bias_epilogue(
                        outputs[i],
                        biases[i],
                        m,
                        projections[i].n,
                        projections[i].n);
                }
            }
            if (PerfStatsCollector::isDomainEnabled("kernel"))
            {
                PerfStatsCollector::addCounter(
                    "kernel",
                    "cpu_native_vnni_small_m_fused_projection_calls",
                    1.0,
                    "gemm",
                    "cpu",
                    PerfStatsCollector::Tags{
                        {"m", std::to_string(m)},
                        {"k", std::to_string(k)},
                        {"projections", std::to_string(projections.size())}});
            }
            return true;
        }

        /**
         * @brief Group MTP verifier rows while preserving M=1 decode GEMV math.
         *
         * This is the CPU implementation of ITensorGemm's Phase 9.8 verifier
         * contract.  It quantizes the M verifier activation rows once, then
         * runs every projection through gemm_native_vnni_preq_decode_equivalent_rows().
         * That helper parallelizes across rows and N/K tasks, but each row uses
         * the same chunk kernels and reduction order as serial decode.  The
         * result is grouped/concurrent execution without the 2-row GEMM
         * accumulation drift that can poison GDN/short-conv state publication.
         */
        bool multiply_fused_verifier_rows_decode_equivalent(
            const TensorBase *input,
            const std::vector<TensorProjectionDesc> &projections,
            int m, int k,
            const IMPIContext *mpi_ctx = nullptr,
            DeviceWorkspaceManager *workspace = nullptr) override
        {
            (void)mpi_ctx;
            (void)workspace;

            constexpr size_t kMaxGroupedProjections = 16u;
            if (!valid_ || !input || m <= 1 || k <= 0 ||
                projections.empty() ||
                projections.size() > kMaxGroupedProjections)
            {
                LOG_ERROR("[CPUNativeVNNIGemmKernel] grouped verifier projection rejected: valid="
                          << valid_ << " input=" << (input != nullptr)
                          << " m=" << m << " k=" << k
                          << " projections=" << projections.size());
                return false;
            }

            const float *input_data = input->data();
            if (!input_data)
            {
                LOG_ERROR("[CPUNativeVNNIGemmKernel] grouped verifier projection rejected: input has no host FP32 data");
                return false;
            }

            std::array<CPUNativeVNNIGemmKernel *, kMaxGroupedProjections>
                vnni_kernels = {};
            for (size_t i = 0; i < projections.size(); ++i)
            {
                const auto &proj = projections[i];
                auto *vnni = dynamic_cast<CPUNativeVNNIGemmKernel *>(proj.kernel);
                if (!vnni || !vnni->valid_ || !proj.output || proj.n <= 0)
                {
                    LOG_ERROR("[CPUNativeVNNIGemmKernel] grouped verifier projection rejected at projection "
                              << i << ": native_vnni=" << (vnni != nullptr)
                              << " valid=" << (vnni ? vnni->valid_ : false)
                              << " output=" << (proj.output != nullptr)
                              << " n=" << proj.n);
                    return false;
                }

                /*
                 * The fused projection API shares one activation transform.
                 * If a future model ever mixes differently rotated weights in
                 * one fused group, that is a real contract gap and must be
                 * handled explicitly rather than silently producing drift.
                 */
                if (vnni->activation_rotation_ != activation_rotation_)
                {
                    LOG_ERROR("[CPUNativeVNNIGemmKernel] grouped verifier projection rejected at projection "
                              << i << ": mixed activation rotation contracts");
                    return false;
                }
                vnni_kernels[i] = vnni;
            }

            const bool perf_enabled =
                PerfStatsCollector::isDomainEnabled("kernel");
            auto perf_start = perf_enabled ? PerfStatsCollector::Clock::now()
                                           : PerfStatsCollector::Clock::time_point{};
            input_data = maybe_rotate_activation(input_data, m, k);

            const int K_blocks = (k + 31) / 32;
            const size_t shared_q8_blocks = static_cast<size_t>(m) * K_blocks;
            thread_local AlignedVector<Q8_1Block> shared_q8_tls;
            if (shared_q8_tls.size() < shared_q8_blocks)
                shared_q8_tls.resize_uninitialized(shared_q8_blocks);
            quantize_activations_to_q8_1(
                input_data,
                shared_q8_tls.data(),
                m,
                k,
                K_blocks,
                numerical_policy_);
            recordVerifierTiming(
                "cpu_native_vnni_verifier_projection_activation_quantize",
                perf_start,
                m,
                /*n=*/0,
                k,
                static_cast<int>(projections.size()));

            /*
             * Multi-projection verifier path.
             *
             * GDN/QKV verifier graphs commonly have several projections fed by
             * the same runtime-M hidden-state rows. The older implementation ran
             * one grouped-row GEMV per projection, which was correct but paid
             * an OpenMP team entry and scheduling cost for every projection.
             * This fused descriptor path keeps the exact M=1 decode chunk math
             * and schedules every projection under one OpenMP team.
             */
            if (projections.size() >= 2)
            {
                std::array<FusedVerifierRowsDesc, kMaxGroupedProjections>
                    fused_descs = {};
                for (size_t i = 0; i < projections.size(); ++i)
                {
                    const auto &proj = projections[i];
                    float *out_data = proj.output->mutable_data();
                    if (!out_data)
                        return false;

                    fused_descs[i] = {
                        &vnni_kernels[i]->packed_,
                        out_data,
                        proj.bias ? proj.bias->data() : nullptr,
                        proj.n,
                        proj.n};
                }

                perf_start = perf_enabled ? PerfStatsCollector::Clock::now()
                                          : PerfStatsCollector::Clock::time_point{};
                if (gemm_native_vnni_fused_verifier_rows_preq(
                        shared_q8_tls.data(),
                        fused_descs.data(),
                        static_cast<int>(projections.size()),
                        m,
                        K_blocks))
                {
                    recordVerifierTiming(
                        "cpu_native_vnni_verifier_projection_fused_gemv",
                        perf_start,
                        m,
                        /*n=*/0,
                        k,
                        static_cast<int>(projections.size()));
                    if (PerfStatsCollector::isDomainEnabled("kernel"))
                    {
                        PerfStatsCollector::addCounter(
                            "kernel",
                            "cpu_native_vnni_fused_grouped_verifier_projection_calls",
                            1.0,
                            "gemm",
                            "cpu",
                            PerfStatsCollector::Tags{
                                {"m", std::to_string(m)},
                                {"k", std::to_string(k)},
                                {"projections", std::to_string(projections.size())}});
                    }
                    return true;
                }

                LOG_ERROR("[CPUNativeVNNIGemmKernel] fused grouped verifier "
                          "kernel rejected an advertised multi-projection "
                          "NativeVNNI contract");
                recordVerifierTiming(
                    "cpu_native_vnni_verifier_projection_fused_gemv_rejected",
                    perf_start,
                    m,
                    /*n=*/0,
                    k,
                    static_cast<int>(projections.size()));

                return false;
            }

            for (size_t i = 0; i < projections.size(); ++i)
            {
                auto *vnni = vnni_kernels[i];
                const auto &proj = projections[i];
                float *out_data = proj.output->mutable_data();
                if (!out_data)
                    return false;

                perf_start = perf_enabled ? PerfStatsCollector::Clock::now()
                                          : PerfStatsCollector::Clock::time_point{};
                gemm_native_vnni_preq_decode_equivalent_rows(
                    vnni->packed_,
                    shared_q8_tls.data(),
                    out_data,
                    m,
                    proj.n);
                recordVerifierTiming(
                    "cpu_native_vnni_verifier_projection_per_gemv",
                    perf_start,
                    m,
                    proj.n,
                    k,
                    1);

                if (proj.bias)
                {
                    const float *bias_data = proj.bias->data();
                    if (!bias_data)
                        return false;
                    apply_bias_epilogue(out_data, bias_data, m, proj.n, proj.n);
                }
            }

            if (PerfStatsCollector::isDomainEnabled("kernel"))
            {
                PerfStatsCollector::addCounter(
                    "kernel",
                    "cpu_native_vnni_grouped_verifier_projection_calls",
                    1.0,
                    "gemm",
                    "cpu",
                    PerfStatsCollector::Tags{
                        {"m", std::to_string(m)},
                        {"k", std::to_string(k)},
                        {"projections", std::to_string(projections.size())}});
            }
            return true;
        }

        /**
         * @brief One independently routed pre-quantized projection.
         *
         * A CPU MoE layer contains many small expert matrices with unequal row
         * counts. Describing each matrix explicitly lets the NativeVNNI
         * scheduler execute the complete layer under one persistent OpenMP
         * team while preserving each row's serial-decode arithmetic contract.
         */
        struct BatchedPrequantizedProjectionDesc
        {
            CPUNativeVNNIGemmKernel *kernel = nullptr; ///< Prepared weight owner.
            const Q8_1Block *input_q8 = nullptr;       ///< Projection-local rows.
            float *output = nullptr;                   ///< Row-major destination.
            const float *bias = nullptr;               ///< Optional projection bias.
            int rows = 0;                              ///< Runtime rows for this expert.
            int n = 0;                                 ///< Logical output columns.
            int ldc = 0;                               ///< Destination row stride.
            /**
             * Grouped-row policy override used by byte-exact certification.
             * Production descriptors retain Auto. One-row projections ignore
             * this field because their independent M=1 policy has a distinct
             * generated authority.
             */
            VerifierRowsPolicy verifier_schedule = VerifierRowsPolicy::Auto;
        };

        /** Maximum gate/up descriptors for one 256-expert MoE layer. */
        static constexpr int kMaxBatchedPrequantizedProjections = 512;

        /**
         * @brief Execute unequal-M pre-quantized projections in one CPU team.
         *
         * Every descriptor independently resolves the sealed M=1/grouped
         * decode policy for its codebook, geometry, ISA, and row count. The
         * underlying launcher shares one persistent OpenMP team across the
         * entire descriptor set, eliminating per-expert team reconstruction.
         * No descriptor is replayed row by row and no alternate arithmetic
         * path is available.
         *
         * @param descriptors Complete projection set for one layer phase.
         * @param descriptor_count Number of valid entries in @p descriptors.
         * @param k Shared logical activation width.
         * @return `true` after every projection completes; `false` when the
         *         explicit eager/unrotated NativeVNNI contract is invalid.
         */
        static bool multiply_batched_preq_decode_equivalent(
            const BatchedPrequantizedProjectionDesc *descriptors,
            int descriptor_count,
            int k)
        {
            if (!descriptors || descriptor_count <= 0 ||
                descriptor_count > kMaxBatchedPrequantizedProjections ||
                k <= 0)
            {
                LOG_ERROR("[CPUNativeVNNIGemmKernel] batched pre-quantized "
                          "projection received invalid geometry: descriptors="
                          << descriptor_count << " k=" << k);
                return false;
            }

            std::array<FusedVerifierRowsDesc,
                       kMaxBatchedPrequantizedProjections>
                fused_descriptors = {};
            int total_rows = 0;
            int max_rows = 0;
            for (int projection = 0;
                 projection < descriptor_count;
                 ++projection)
            {
                const auto &source = descriptors[projection];
                CPUNativeVNNIGemmKernel *kernel = source.kernel;
                if (!kernel || !kernel->valid_ || !source.input_q8 ||
                    !source.output || source.rows <= 0 || source.n <= 0 ||
                    source.ldc < source.n || kernel->packed_.K != k ||
                    kernel->packed_.N != source.n ||
                    kernel->activation_rotation_ != nullptr)
                {
                    LOG_ERROR("[CPUNativeVNNIGemmKernel] batched pre-quantized "
                              "projection contract mismatch at descriptor "
                              << projection);
                    return false;
                }

                fused_descriptors[static_cast<size_t>(projection)] = {
                    .packed = &kernel->packed_,
                    .output = source.output,
                    .bias = source.bias,
                    .N = source.n,
                    .ldc = source.ldc,
                    .rows = source.rows,
                    .input = source.input_q8,
                    .decode_schedule = DecodeSchedulePolicy::Auto,
                    .verifier_schedule = source.rows > 1
                        ? source.verifier_schedule
                        : VerifierRowsPolicy::Auto,
                };
                total_rows += source.rows;
                max_rows = std::max(max_rows, source.rows);
            }

            const bool team_observer =
                !omp_in_parallel() || omp_get_thread_num() == 0;
            const bool perf_enabled =
                team_observer &&
                PerfStatsCollector::isDomainEnabled("kernel");
            const auto perf_start = perf_enabled
                                        ? PerfStatsCollector::Clock::now()
                                        : PerfStatsCollector::Clock::time_point{};
            const int blocks_per_row =
                (k + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
            if (!gemm_native_vnni_fused_verifier_rows_preq(
                    nullptr,
                    fused_descriptors.data(),
                    descriptor_count,
                    /*default_rows unused by explicit descriptors=*/1,
                    blocks_per_row))
            {
                return false;
            }

            if (team_observer)
            {
                recordVerifierTiming(
                    "cpu_native_vnni_batched_preq_decode_equivalent",
                    perf_start,
                    total_rows,
                    descriptor_count,
                    k,
                    descriptor_count);
                PerfStatsCollector::addCounter(
                    "kernel",
                    "cpu_native_vnni_batched_preq_decode_equivalent_calls",
                    1.0,
                    "gemm",
                    "cpu",
                    {{"descriptors", std::to_string(descriptor_count)},
                     {"total_rows", std::to_string(total_rows)},
                     {"max_rows", std::to_string(max_rows)},
                     {"k", std::to_string(k)},
                     {"team_lifetime", "layer_phase"}});
            }
            return true;
        }

        /**
         * @brief Execute one complete grouped CPU MoE FFN in a stable team.
         *
         * Sparse routed experts expose three dependent parallel phases. Starting
         * a fresh OpenMP region for gate/up, activation publication, and down
         * spends a material fraction of verifier latency in libgomp handoff and
         * completion barriers. This transaction makes the phase DAG explicit
         * and keeps one physical-core team alive across all three operations.
         *
         * Every worker enters every nested-safe workshare. Gate/up completion
         * therefore happens before the exact fused SwiGLU-to-Q8_1 publication,
         * and that publication completes before down projection begins. No
         * arithmetic is fused across those boundaries: each row retains the
         * same Q8 blocks, FP32 accumulation order, and rounding points as the
         * ordinary serial-decode-equivalent kernels.
         *
         * @param gate_up_descriptors Gate and up projections for all local experts.
         * @param gate_up_count Number of valid gate/up descriptors.
         * @param hidden_width Logical K dimension of gate/up matrices.
         * @param gate_rows Contiguous gate projection output rows.
         * @param up_rows Contiguous up projection output rows.
         * @param activation_q8 Destination for exact Q8_1 SwiGLU publication.
         * @param route_rows Total expert-major route rows in the transaction.
         * @param intermediate Expert hidden width and SwiGLU row width.
         * @param activation_blocks_per_row Q8_1 blocks in one SwiGLU row.
         * @param down_descriptors Down projections for all local experts.
         * @param down_count Number of valid down descriptors.
         * @return `true` only when every team participant completes every phase.
         */
        static bool execute_moe_grouped_ffn_transaction_preq_decode_equivalent(
            const BatchedPrequantizedProjectionDesc *gate_up_descriptors,
            int gate_up_count,
            int hidden_width,
            const float *gate_rows,
            const float *up_rows,
            Q8_1Block *activation_q8,
            int route_rows,
            int intermediate,
            int activation_blocks_per_row,
            const BatchedPrequantizedProjectionDesc *down_descriptors,
            int down_count)
        {
            if (!gate_up_descriptors || gate_up_count <= 0 ||
                !down_descriptors || down_count <= 0 ||
                !gate_rows || !up_rows || !activation_q8 ||
                hidden_width <= 0 || route_rows <= 0 || intermediate <= 0 ||
                activation_blocks_per_row <= 0)
            {
                LOG_ERROR("[CPUNativeVNNIGemmKernel] MoE verifier transaction "
                          "received an incomplete descriptor or buffer contract");
                return false;
            }

            auto validate_bundle = [](const BatchedPrequantizedProjectionDesc *descriptors,
                                      int count,
                                      int expected_k,
                                      const char *phase) -> bool
            {
                if (count > kMaxBatchedPrequantizedProjections)
                    return false;
                for (int projection = 0; projection < count; ++projection)
                {
                    const auto &source = descriptors[projection];
                    CPUNativeVNNIGemmKernel *kernel = source.kernel;
                    if (!kernel || !kernel->valid_ || !source.input_q8 ||
                        !source.output || source.rows <= 0 || source.n <= 0 ||
                        source.ldc < source.n ||
                        kernel->packed_.K != expected_k ||
                        kernel->packed_.N != source.n ||
                        kernel->activation_rotation_ != nullptr)
                    {
                        LOG_ERROR("[CPUNativeVNNIGemmKernel] MoE grouped FFN "
                                  << phase << " descriptor " << projection
                                  << " violates the eager pre-quantized contract");
                        return false;
                    }
                }
                return true;
            };
            if (!validate_bundle(
                    gate_up_descriptors,
                    gate_up_count,
                    hidden_width,
                    "gate/up") ||
                !validate_bundle(
                    down_descriptors,
                    down_count,
                    intermediate,
                    "down"))
            {
                return false;
            }

            std::atomic<bool> transaction_ok{true};
            auto execute_participant = [&]()
            {
                if (!multiply_batched_preq_decode_equivalent(
                        gate_up_descriptors,
                        gate_up_count,
                        hidden_width))
                {
                    transaction_ok.store(false, std::memory_order_relaxed);
                }

                swiglu_quantize_activations_to_q8_1(
                    gate_rows,
                    up_rows,
                    activation_q8,
                    route_rows,
                    intermediate,
                    activation_blocks_per_row,
                    gate_up_descriptors[0].kernel->numerical_policy_);

                if (!multiply_batched_preq_decode_equivalent(
                        down_descriptors,
                        down_count,
                        intermediate))
                {
                    transaction_ok.store(false, std::memory_order_relaxed);
                }
            };
            OMP_WORKSHARE_REGION(execute_participant);

            const bool completed =
                transaction_ok.load(std::memory_order_relaxed);
            if (!completed)
            {
                LOG_ERROR("[CPUNativeVNNIGemmKernel] MoE verifier transaction "
                          "failed inside its stable OpenMP team");
                return false;
            }

            if (!omp_in_parallel() || omp_get_thread_num() == 0)
            {
                PerfStatsCollector::addCounter(
                    "kernel",
                    "cpu_native_vnni_moe_grouped_ffn_transactions",
                    1.0,
                    "gemm",
                    "cpu",
                    {{"gate_up_descriptors", std::to_string(gate_up_count)},
                     {"down_descriptors", std::to_string(down_count)},
                     {"route_rows", std::to_string(route_rows)},
                     {"hidden_width", std::to_string(hidden_width)},
                     {"intermediate", std::to_string(intermediate)},
                     {"team_lifetime", "complete_moe_ffn"}});
            }
            return true;
        }

        /**
         * @brief Project router-published Q8_1 rows without requantizing hidden state.
         *
         * Routed MoE prefill and verifier execution group route slots by expert.
         * A single hidden row can therefore appear in several expert batches;
         * quantizing each batch independently is both redundant and a source of
         * accidental divergence from serial decode. The CPU router publishes
         * the canonical Q8_1 row set once, and this method consumes gathered
         * blocks from that publication directly.
         *
         * M=1 batches use the ordinary fused decode GEMV kernel. Multi-row
         * batches use the grouped decode-equivalent kernel, whose per-row K
         * reduction order is identical to M=1 decode. The method has no FP32
         * alternative: every projection must be an eager, unrotated
         * NativeVNNI kernel with a compatible matrix shape.
         *
         * @param input_q8 Contiguous Q8_1 rows in `[m, ceil(k / 32)]` layout.
         * @param projections Gate/up projection bundle sharing the same rows.
         * @param m Number of gathered route rows. Runtime M is bounded only by
         *          the caller's declared graph and scratch capacity.
         * @param k FP32 logical width represented by each Q8_1 row.
         */
        bool multiply_fused_router_q8_hidden_grouped_decode_equivalent(
            const Q8_1Block *input_q8,
            const std::vector<TensorProjectionDesc> &projections,
            int m,
            int k)
        {
            constexpr size_t kMaxRouterQ8Projections = 16u;
            if (!valid_ || !input_q8 || m < 1 || k <= 0 ||
                projections.empty() ||
                projections.size() > kMaxRouterQ8Projections)
            {
                LOG_ERROR("[CPUNativeVNNIGemmKernel] router-Q8 projection rejected: valid="
                          << valid_ << " input_q8=" << (input_q8 != nullptr)
                          << " m=" << m << " k=" << k
                          << " projections=" << projections.size());
                return false;
            }

            const int K_blocks = (k + Q8_1Block::BLOCK_SIZE - 1) /
                                 Q8_1Block::BLOCK_SIZE;
            std::array<CPUNativeVNNIGemmKernel *, kMaxRouterQ8Projections>
                vnni_kernels = {};
            for (size_t i = 0; i < projections.size(); ++i)
            {
                const auto &projection = projections[i];
                auto *vnni = dynamic_cast<CPUNativeVNNIGemmKernel *>(projection.kernel);
                if (!vnni || !vnni->valid_ || !projection.output ||
                    projection.n <= 0 || vnni->packed_.K != k ||
                    vnni->packed_.N != projection.n)
                {
                    LOG_ERROR("[CPUNativeVNNIGemmKernel] router-Q8 projection "
                              "contract mismatch at index "
                              << i);
                    return false;
                }
                if (vnni->activation_rotation_ != nullptr)
                {
                    LOG_ERROR("[CPUNativeVNNIGemmKernel] router-Q8 publication "
                              "cannot feed a rotated activation contract at projection "
                              << i);
                    return false;
                }
                vnni_kernels[i] = vnni;
            }

            const bool perf_enabled =
                PerfStatsCollector::isDomainEnabled("kernel");
            const auto perf_start = perf_enabled
                                        ? PerfStatsCollector::Clock::now()
                                        : PerfStatsCollector::Clock::time_point{};

            if (m == 1)
            {
                std::array<FusedGemvDesc, kMaxRouterQ8Projections>
                    descriptors = {};
                for (size_t i = 0; i < projections.size(); ++i)
                {
                    const auto &projection = projections[i];
                    auto &descriptor = descriptors[i];
                    descriptor.packed = &vnni_kernels[i]->packed_;
                    descriptor.output = projection.output->mutable_data();
                    descriptor.bias = projection.bias ? projection.bias->data() : nullptr;
                    descriptor.N = projection.n;
                    if (!descriptor.output || (projection.bias && !descriptor.bias))
                        return false;
                }
                gemv_native_vnni_fused_preq(
                    input_q8,
                    descriptors.data(),
                    static_cast<int>(projections.size()));
            }
            else
            {
                std::array<FusedVerifierRowsDesc, kMaxRouterQ8Projections>
                    descriptors = {};
                for (size_t i = 0; i < projections.size(); ++i)
                {
                    const auto &projection = projections[i];
                    float *output = projection.output->mutable_data();
                    const float *bias = projection.bias ? projection.bias->data() : nullptr;
                    if (!output || (projection.bias && !bias))
                        return false;
                    descriptors[i] = {
                        &vnni_kernels[i]->packed_,
                        output,
                        bias,
                        projection.n,
                        projection.n};
                }

                if (!gemm_native_vnni_fused_verifier_rows_preq(
                        input_q8,
                        descriptors.data(),
                        static_cast<int>(projections.size()),
                        m,
                        K_blocks))
                {
                    LOG_ERROR("[CPUNativeVNNIGemmKernel] router-Q8 fused verifier "
                              "kernel rejected an advertised NativeVNNI contract");
                    return false;
                }
            }

            recordVerifierTiming(
                "cpu_native_vnni_router_q8_reused_projection_gemv",
                perf_start,
                m,
                /*n=*/0,
                k,
                static_cast<int>(projections.size()));
            PerfStatsCollector::addCounter(
                "kernel",
                "cpu_native_vnni_router_q8_grouped_decode_equivalent_projection_calls",
                1.0,
                "gemm",
                "cpu",
                {{"m", std::to_string(m)},
                 {"k", std::to_string(k)},
                 {"projections", std::to_string(projections.size())},
                 {"path", m == 1 ? "decode" : "grouped_rows"}});
            return true;
        }

        // =====================================================================
        // Fused multi-input GEMV for MoE expert down projections
        // =====================================================================

        bool multiply_fused_expert_down(
            const FusedExpertDownDesc *descs, int num_descs,
            int m, int k) override
        {
            if (!descs || m != 1 || num_descs < 1 ||
                num_descs > kMaxBatchedPrequantizedProjections || k <= 0)
                return false;

            // Verify all kernels are CPUNativeVNNIGemmKernel
            for (int i = 0; i < num_descs; ++i)
            {
                auto *vnni = dynamic_cast<CPUNativeVNNIGemmKernel *>(descs[i].kernel);
                if (!vnni || !vnni->valid_ || !descs[i].input ||
                    !descs[i].output || descs[i].n <= 0 ||
                    vnni->packed_.K != k || vnni->packed_.N != descs[i].n ||
                    vnni->activation_rotation_ != nullptr)
                    return false;
            }

            const int K_blocks = (k + 31) / 32;

            // Quantize each expert's FP32 input to Q8_1
            // Use a contiguous buffer for all experts' Q8_1 blocks
            thread_local AlignedVector<Q8_1Block> multi_q8_tls;
            const size_t total_blocks = static_cast<size_t>(num_descs) * K_blocks;
            if (multi_q8_tls.size() < total_blocks)
                multi_q8_tls.resize_uninitialized(total_blocks);

            for (int i = 0; i < num_descs; ++i)
            {
                auto *vnni = static_cast<CPUNativeVNNIGemmKernel *>(
                    descs[i].kernel);
                Q8_1Block *A_q8 = multi_q8_tls.data() + static_cast<size_t>(i) * K_blocks;
                const float *input_data = descs[i].input;
                int kb = 0;
#if defined(__AVX512F__)
                for (; kb + 1 < K_blocks; kb += 2)
                    quantizeTwoActivationBlocks(
                        input_data + kb * 32,
                        A_q8[kb],
                        A_q8[kb + 1],
                        vnni->numerical_policy_);
#endif
                for (; kb < K_blocks; ++kb)
                    quantizeActivationBlock(
                        input_data + kb * 32,
                        A_q8[kb],
                        std::min(32, k - kb * 32),
                        vnni->numerical_policy_);
            }

            std::array<FusedGemvMultiInputDesc,
                       kMaxBatchedPrequantizedProjections>
                mi_descs = {};
            for (int i = 0; i < num_descs; ++i)
            {
                auto *vnni = static_cast<CPUNativeVNNIGemmKernel *>(descs[i].kernel);
                auto &d = mi_descs[static_cast<size_t>(i)];
                d.A_q8 = multi_q8_tls.data() + static_cast<size_t>(i) * K_blocks;
                d.packed = &vnni->packed_;
                d.output = descs[i].output;
                d.N = descs[i].n;
            }

            // Single OMP region with nowait between expert projections
            gemv_fused_multi_input_preq(
                mi_descs.data(),
                num_descs);

            if (PerfStatsCollector::isDomainEnabled("kernel"))
            {
                PerfStatsCollector::addCounter(
                    "kernel",
                    "cpu_native_vnni_fused_expert_down_calls",
                    1.0,
                    "gemm",
                    "cpu",
                    PerfStatsCollector::Tags{
                        {"m", std::to_string(m)},
                        {"k", std::to_string(k)},
                        {"experts", std::to_string(num_descs)}});
            }

            return true;
        }

    private:
        CPUNativeVNNIPackedWeights packed_;
        bool valid_ = false;

        /** Arithmetic identity retained across detach/attach lifecycle events. */
        CPUProjectionNumericalPolicy numerical_policy_ =
            CPUProjectionNumericalPolicy::BackendNative;

        /**
         * @brief Optional block-diagonal activation transform paired with weights.
         *
         * The pointer is immutable after construction and is shared safely by
         * concurrent callers. Scratch storage remains caller-thread local.
         */
        const ActivationRotation *activation_rotation_ = nullptr;

        /// Apply rotation to FP32 activation data, returns pointer to rotated data.
        /// If no rotation is configured, returns the original pointer unchanged.
        const float *maybe_rotate_activation(const float *input, int m, int k) const
        {
            if (!activation_rotation_)
                return input;

            const size_t len = static_cast<size_t>(m) * k;
            thread_local AlignedVector<float> rotation_scratch_tls;
            if (rotation_scratch_tls.size() < len)
                rotation_scratch_tls.resize_uninitialized(len);

            std::memcpy(rotation_scratch_tls.data(), input, len * sizeof(float));
            activation_rotation_->rotate_rows_inplace(rotation_scratch_tls.data(), m, k);
            return rotation_scratch_tls.data();
        }

        /**
         * @brief Record one coarse verifier-kernel timing sample.
         *
         * These timers intentionally sit at the projection bundle boundary, not
         * inside the NativeVNNI block kernels.  That keeps ordinary inference
         * cheap while still telling us whether Phase 9.8 verifier time is spent
         * in activation preparation, grouped projection GEMV, or SwiGLU.
         */
        static void recordVerifierTiming(
            const char *name,
            PerfStatsCollector::Clock::time_point start,
            int m,
            int n,
            int k,
            int projections)
        {
            if (!PerfStatsCollector::isDomainEnabled("kernel"))
                return;

            const auto end = PerfStatsCollector::Clock::now();
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            PerfStatsCollector::Tags tags{
                {"m", std::to_string(m)},
                {"k", std::to_string(k)},
                {"projections", std::to_string(projections)}};
            if (n > 0)
                tags.emplace("n", std::to_string(n));
            PerfStatsCollector::recordTimingNs(
                "kernel",
                name ? name : "cpu_native_vnni_verifier_unknown",
                ns > 0 ? static_cast<uint64_t>(ns) : 0,
                "gemm",
                "cpu",
                std::move(tags));
        }
    };

} // namespace llaminar2::cpu::native_vnni
