/**
 * @file ROCmQuantisedGemmKernelDispatch.cpp
 * @brief ROCmQuantisedGemmKernel ITensorGemm single and fused-tensor dispatch.
 */
#include "ROCmQuantisedGemmKernelInternal.h"

namespace llaminar2
{
    namespace rocm
    {
        // =====================================================================
        // ITensorGemm interface - Implementation
        // =====================================================================

        bool ROCmQuantisedGemmKernel::multiply_tensor(
            const TensorBase *A, TensorBase *C,
            bool transpose_B,
            float alpha, float beta,
            const TensorBase *bias,
            const IMPIContext *mpi_ctx,
            int device_idx,
            DeviceWorkspaceManager *workspace,
            int activation_row_offset)
        {
            (void)mpi_ctx;
            (void)device_idx;
            (void)transpose_B; // Weights are always transposed for this kernel

            if (!A || !C)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Null tensor");
                return false;
            }

            // Get dimensions
            int m = static_cast<int>(A->rows());
            int k = static_cast<int>(A->cols());
            int n = static_cast<int>(N_);

            if (static_cast<size_t>(k) != K_)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] K mismatch: A.cols=" << k << " vs weight K=" << K_);
                return false;
            }

            return multiply_tensor(A, C, m, n, k, transpose_B, alpha, beta, bias, mpi_ctx, device_idx, workspace, activation_row_offset);
        }

        bool ROCmQuantisedGemmKernel::multiply_tensor(
            const TensorBase *A, TensorBase *C,
            int m, int n, int k,
            bool transpose_B,
            float alpha, float beta,
            const TensorBase *bias,
            const IMPIContext *mpi_ctx,
            int device_idx,
            DeviceWorkspaceManager *workspace,
            int activation_row_offset)
        {
            // Decode work admission never changes the selected arithmetic program.
            if (m > 1 && currentProjectionRowWork(m) && !VerifierKernelModeScope::active())
                throw std::invalid_argument("Projection row intervals require explicit grouped Decode arithmetic for M>1");

            if (llaminar2::rocm::currentNativeQ8SerialGeometry())
            {
                // A geometry experiment must reach the actual native-Q8 M1
                // producer. Reject other prepared families/operations before
                // they can enqueue a byte-equal but unmeasured substitute.
                llaminar2::rocm::clearNativeQ8SerialLaunch();
                llaminar2::rocm::validateNativeQ8SerialScopeBinding(rocm_device_id_, gpu_stream_);
                if (m != 1 || n <= 0 || k <= 0 || alpha != 1.0f ||
                    beta != 0.0f || bias || !impl_ || !impl_->has_native_quantised ||
                    impl_->native_quantised_codebook_id != 19 ||
                    !impl_->d_weights_native_quantised || !impl_->d_weights_native_scales)
                    throw std::logic_error("native-Q8 serial geometry scope requires the prepared native19 M1 alpha1/beta0 projection");
            }
            auto execution_stream =
                static_cast<hipStream_t>(requireGPUStream());
            ROCM_KERNEL_PROFILE_SCOPE_STREAM(
                ROCmKernelType::GEMM, execution_stream);
            (void)mpi_ctx;
            (void)device_idx;
            (void)transpose_B;

            // Use passed workspace if provided, otherwise fall back to bound workspace
            DeviceWorkspaceManager *ws = workspace ? workspace : workspace_;
            DeviceWorkspaceManager *saved_workspace = workspace_;
            if (ws && ws != workspace_)
            {
                // Temporarily bind passed workspace for this call
                workspace_ = ws;
            }
            auto restore_workspace = [&](void *)
            {
                if (ws && ws != saved_workspace)
                {
                    workspace_ = saved_workspace;
                }
            };
            std::unique_ptr<void, decltype(restore_workspace)> workspace_restore_guard(
                (ws && ws != saved_workspace) ? static_cast<void *>(this) : nullptr,
                restore_workspace);

            if (!A || !C)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Null tensor");
                return false;
            }

            if (!validateROCmWorkspaceBinding(ws, rocm_device_id_, "ROCmQuantisedGemmKernel::multiply_tensor"))
            {
                return false;
            }

            if (m <= 0 || n <= 0 || k <= 0)
            {
                LOG_WARN("[ROCmQuantisedGemmKernel::multiply_tensor] Zero dimensions");
                return true;
            }

            // Cast to FP32Tensor (only supported type for now)
            auto *A_fp32 = dynamic_cast<const FP32Tensor *>(A);
            auto *C_fp32 = dynamic_cast<FP32Tensor *>(C);

            if (!A_fp32)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] A must be FP32Tensor (Q8_1 not yet supported)");
                return false;
            }
            if (!C_fp32)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] C must be FP32Tensor");
                return false;
            }

            /*
             * Tensor kernels never own placement. The graph executor or a
             * direct-kernel harness prepares storage first; this boundary only
             * validates exact residency and joins the producer event.
             */
            const DeviceId target_device =
                DeviceId::rocm(rocm_device_id_);
            TransferEngine::requireDeviceInput(
                const_cast<FP32Tensor *>(A_fp32),
                target_device,
                gpu_stream_);
            if (beta != 0.0f)
                TransferEngine::requireDeviceInput(
                    C_fp32, target_device, gpu_stream_);
            else
                TransferEngine::requireDeviceOutput(
                    C_fp32, target_device, gpu_stream_);

            // Check if tensors are on GPU
            const float *d_input = static_cast<const float *>(A_fp32->gpu_data_ptr());
            float *d_output = static_cast<float *>(C_fp32->gpu_data_ptr());

            // Apply activation row offset (used by LMHeadStage to process only last token)
            if (activation_row_offset > 0 && d_input != nullptr)
            {
                d_input += static_cast<size_t>(activation_row_offset) * k;
                LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Applied activation_row_offset="
                          << activation_row_offset << " (offset " << (static_cast<size_t>(activation_row_offset) * k) << " floats)");
            }

            const bool use_gpu_path = (d_input != nullptr) && (d_output != nullptr);
            if (!use_gpu_path)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] GPU execution requires device-resident input and output tensors");
                return false;
            }

            // Gate legacy phase timing behind LLAMINAR_PROFILE_KERNELS to eliminate ~18 chrono
            // calls per GEMM (~630ns × 140 GEMMs/token = ~88μs/token).
            const bool phase_timing = debugEnv().profile.enabled;
            std::chrono::high_resolution_clock::time_point phase_start{};
            std::chrono::high_resolution_clock::time_point phase_end{};

            {
                LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Using GPU-to-GPU path (d_input="
                          << d_input << ", d_output=" << d_output << ")");

                if (!validatePointerDeviceOrLog(
                        d_input, rocm_device_id_, "d_input", "ROCmQuantisedGemmKernel::multiply_tensor") ||
                    !validatePointerDeviceOrLog(
                        d_output, rocm_device_id_, "d_output", "ROCmQuantisedGemmKernel::multiply_tensor"))
                {
                    return false;
                }
            }

            // =========================================================================
            // DISPATCH POLICY (unified):
            //
            //   ≤6-bit weights (has_native_quantised) → native-quantised GEMV/GEMM
            //   8-bit weights (d_weights_int8_vnni) → INT8-VNNI GEMV/GEMM
            //
            // If both VNNI paths fail, that is a hard error.
            // =========================================================================

            // =========================================================================
            // DECODE FAST PATH: M=1 GEMV (skips activation quantization + CK GEMM)
            //
            // For M=1 (single-token decode), the CK INT8 GEMM is catastrophically
            // inefficient (2.4 TFLOPS = 8% of MI60 peak) because decode is
            // bandwidth-bound, not compute-bound. Our custom GEMV kernel reads
            // INT8 weights directly and multiplies with FP32 activations,
            // eliminating:
            //   1. FP32→INT8 activation quantization kernel
            //   2. M-padding (hipMemset + hipMemcpy)
            //   3. CK 32×32 tile overhead (wastes 31 of 32 rows)
            //   4. INT32→FP32 scale application kernel
            //
            // Handles optional bias in a single fused kernel launch.
            // =========================================================================
            if (g_rocm_native_quantised_decode_equivalent_scope &&
                (m > 1 || countedVerifierRows(m)))
            {
                ensureWeightsConverted();
            if (currentProjectionRowWork(m) && (!impl_ || !impl_->has_native_quantised))
                throw std::invalid_argument("Decode projection work requires the prepared all-codebook NativeQuantised route");

                int8_t *d_weights_vnni = impl_ ? impl_->d_weights_int8_vnni : nullptr;
                float *d_scales_B = nullptr;
                if (packed_)
                {
                    d_scales_B = packed_->d_scales;
                }
                else if (impl_)
                {
                    d_scales_B = impl_->d_scales_B;
                }

                if (!impl_ || (!impl_->has_native_quantised && !(d_weights_vnni && d_scales_B)))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Small-M verifier path has no supported GEMV weights");
                    return false;
                }

                const float *d_bias = nullptr;
                if (bias)
                {
                    auto *bias_tensor = const_cast<TensorBase *>(bias);
                    const auto target_device =
                        DeviceId::rocm(rocm_device_id_);
                    TransferEngine::requireDeviceInput(
                        bias_tensor, target_device, gpu_stream_);

                    d_bias = static_cast<const float *>(bias->gpu_data_ptr());
                }

                validateWorkspace();

                if (d_weights_vnni &&
                    !validatePointerDeviceOrLog(
                        d_weights_vnni, rocm_device_id_, "d_weights_vnni", "ROCmQuantisedGemmKernel::multiply_tensor"))
                {
                    return false;
                }
                if (d_scales_B && !validatePointerDeviceOrLog(
                                      d_scales_B, rocm_device_id_, "d_scales_B", "ROCmQuantisedGemmKernel::multiply_tensor"))
                {
                    return false;
                }
                if (!validatePointerDeviceOrLog(
                        impl_->d_A_int8, rocm_device_id_, "workspace::QUANT_A", "ROCmQuantisedGemmKernel::multiply_tensor") ||
                    !validatePointerDeviceOrLog(
                        impl_->d_scales_A, rocm_device_id_, "workspace::SCALES_A", "ROCmQuantisedGemmKernel::multiply_tensor") ||
                    !validatePointerDeviceOrLog(
                        impl_->d_C_int32, rocm_device_id_, "workspace::ACC_INT32", "ROCmQuantisedGemmKernel::multiply_tensor"))
                {
                    return false;
                }

                const bool gemv_output_is_mapped = C_fp32->isMapped();
                const bool gemv_output_needs_copyout = gemv_output_is_mapped;

                LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Small-M GEMV verifier path M="
                          << m << " N=" << n << " K=" << k
                          << (d_bias ? " +bias" : ""));


                const bool supports_native_small_m =
                    impl_->has_native_quantised && (m >= 2 || countedVerifierRows(m));
                if (supports_native_small_m)
                {
                    // Device-local C is owned by the terminal producer. Mapped C
                    // retains a coalesced row-admitted publication of the raw bank.
                    const auto destination_fold = gemv_output_is_mapped
                        ? ROCmFP32DestinationFold{}
                        : ROCmFP32DestinationFold{d_bias, alpha, beta};
                    float *grouped_output = gemv_output_needs_copyout ? impl_->d_C_fp32 : d_output;
                    if (gemv_output_needs_copyout &&
                        (!grouped_output || grouped_output == d_output ||
                         workspace_->getBufferSize(GemmWorkspaceBuffers::TEMP_C_FP32) <
                             static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float)))
                        throw std::invalid_argument("Grouped verifier epilogue requires distinct full-capacity persistent FP32 scratch");
                    if ((k % 32) != 0)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] NativeQuantised small-M verifier requires K multiple of 32, got M="
                                  << m << " K=" << k);
                        return false;
                    }

                    if (!rocmQuantGemm_quantizeActivationsBlockwiseWithSumsScoped(
                            d_input,
                            impl_->d_A_int8,
                            impl_->d_scales_A_blockwise,
                            impl_->d_sums_A_blockwise,
                            m,
                            k,
                            rocm_device_id_,
                            gpu_stream_))
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] NativeQuantised small-M blockwise activation quantization failed");
                        return false;
                    }

                    if (!rocmGemv_native_quantised_small_m_fp32_with_sums_policy(
                            impl_->d_A_int8,
                            impl_->d_weights_native_quantised,
                            impl_->d_weights_native_scales,
                            impl_->d_weights_native_mins,
                            impl_->d_weights_native_emins,
                            grouped_output,
                            impl_->d_scales_A_blockwise,
                            impl_->d_sums_A_blockwise,
                            impl_->d_scatter_partial,
                            m, n, k,
                            impl_->native_quantised_codebook_id,
                            requireNativeQuantisedArithmeticPolicyCodebook(
                                impl_->native_quantised_codebook_id,
                                impl_->native_source_identity),
                            rocm_device_id_, gpu_stream_,
                            VerifierKernelModeScope::rowsFor(m), destination_fold))
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] NativeQuantised small-M verifier GEMV failed");
                        return false;
                    }

                    if (gemv_output_needs_copyout)
                    {
                        if (!rocmQuantGemm_applyFp32CombineEpilogue(
                                grouped_output, d_output, beta != 0.0f ? d_output : nullptr,
                                d_bias, m, n, alpha, beta, rocm_device_id_, gpu_stream_,
                                VerifierKernelModeScope::rowsFor(m)))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Grouped verifier combine/copy failed");
                            return false;
                        }
                    }

                    if (PerfStatsCollector::isDomainEnabled("kernel"))
                    {
                        PerfStatsCollector::addCounter(
                            "kernel",
                            "rocm_native_quantised_small_m_calls",
                            1.0,
                            "gemm",
                            "rocm:" + std::to_string(rocm_device_id_),
                            PerfStatsCollector::Tags{
                                {"m", std::to_string(m)},
                                {"codebook", std::to_string(static_cast<int>(impl_->native_quantised_codebook_id))},
                                {"n", std::to_string(n)},
                                {"k", std::to_string(k)}});

                    }

            if (PerfStatsCollector::isDomainEnabled("kernel"))
                    {
                        PerfStatsCollector::addCounter(
                            "kernel",
                            "rocm_native_quantised_small_m_rows",
                            static_cast<double>(m),
                            "gemm",
                            "rocm:" + std::to_string(rocm_device_id_),
                            PerfStatsCollector::Tags{
                                {"m", std::to_string(m)},
                                {"codebook", std::to_string(static_cast<int>(impl_->native_quantised_codebook_id))},
                                {"n", std::to_string(n)},
                                {"k", std::to_string(k)}});
                    }

                    publishROCmTensorWrite(*C, rocm_device_id_, gpu_stream_);
                    return true;
                }

                throw std::invalid_argument(
                    "Grouped ROCm verifier projection requires prepared NativeQuantised weights");
            }

            if (m == 1 && !countedVerifierRows(m))
            {
                // Ensure weights are on device
                ensureWeightsConverted();
            if (currentProjectionRowWork(m) && (!impl_ || !impl_->has_native_quantised))
                throw std::invalid_argument("Decode projection work requires the prepared all-codebook NativeQuantised route");

                // Option B: weights are stored as VNNI on device
                int8_t *d_weights_vnni = impl_ ? impl_->d_weights_int8_vnni : nullptr;
                float *d_scales_B = nullptr;
                if (packed_)
                {
                    d_scales_B = packed_->d_scales;
                }
                else if (impl_)
                {
                    d_scales_B = impl_->d_scales_B;
                }

                if ((impl_ && impl_->has_native_quantised) || (d_weights_vnni && d_scales_B))
                {
                    // Resolve bias device pointer if present
                    const float *d_bias = nullptr;
                    if (bias)
                    {
                        auto *bias_tensor = const_cast<TensorBase *>(bias);
                        const auto target_device =
                            DeviceId::rocm(rocm_device_id_);
                        TransferEngine::requireDeviceInput(
                            bias_tensor, target_device, gpu_stream_);

                        d_bias = static_cast<const float *>(bias->gpu_data_ptr());
                    }

                    LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] GEMV fast path M=1 N=" << n << " K=" << k
                                                                                                 << (d_bias ? " +bias" : ""));

                    // INT8 VNNI GEMV (only path — fp16/fp32 modes removed)
                    if (!impl_)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] INT8 GEMV requires impl_ buffers");
                        return false;
                    }

                    validateWorkspace();

                    // =================================================================
                    // CRITICAL: Detect mapped output memory (same fix as prefill).
                    // Mapped memory is used for logits (CPU sampling needs host access).
                    // GPU scattered writes to mapped memory go over PCIe (~1.6 GB/s)
                    // instead of HBM (~1000 GB/s). For LM_Head M=1 N=152064 → 608KB,
                    // this causes ~2700µs vs ~520µs expected from HBM bandwidth.
                    // Fix: redirect kernel output to HBM workspace, then bulk DMA.
                    // =================================================================
                    const bool gemv_output_is_mapped = C_fp32->isMapped();
                    // A native beta GEMV keeps its raw product until the terminal fold.
                    float *const d_gemv_output = (gemv_output_is_mapped || beta != 0.0f)
                        ? impl_->d_C_fp32 : d_output;
                    if (gemv_output_is_mapped)
                    {
                        static std::once_flag gemv_mapped_once;
                        std::call_once(gemv_mapped_once, [&]()
                                       { LOG_WARN("[multiply_tensor] GEMV MAPPED OUTPUT REDIRECT: M=1 N=" << n
                                                                                                          << " mapped_ptr=" << d_output
                                                                                                          << " -> d_C_fp32=" << impl_->d_C_fp32
                                                                                                          << " (" << (n * 4 / 1024) << " KB)"); });
                    }

                    if (d_weights_vnni &&
                        !validatePointerDeviceOrLog(
                            d_weights_vnni, rocm_device_id_, "d_weights_vnni", "ROCmQuantisedGemmKernel::multiply_tensor"))
                    {
                        return false;
                    }
                    // d_scales_B is only used by the INT8-VNNI path, not native-quantised.
                    // NativeQuantised weights use impl_->d_weights_native_scales instead.
                    if (d_scales_B && !validatePointerDeviceOrLog(
                                          d_scales_B, rocm_device_id_, "d_scales_B", "ROCmQuantisedGemmKernel::multiply_tensor"))
                    {
                        return false;
                    }
                    if (!validatePointerDeviceOrLog(
                            impl_->d_A_int8, rocm_device_id_, "workspace::QUANT_A", "ROCmQuantisedGemmKernel::multiply_tensor") ||
                        !validatePointerDeviceOrLog(
                            impl_->d_scales_A, rocm_device_id_, "workspace::SCALES_A", "ROCmQuantisedGemmKernel::multiply_tensor") ||
                        !validatePointerDeviceOrLog(
                            impl_->d_C_int32, rocm_device_id_, "workspace::ACC_INT32", "ROCmQuantisedGemmKernel::multiply_tensor"))
                    {
                        return false;
                    }

                    // =====================================================================
                    // NativeQuantised path: lossless Q4/IQ4 decode, typed block metadata, scale_A inline.
                    // Native GEMV produces FP32, then applies alpha/beta/bias with the shared epilogue when needed.
                    // =====================================================================
                    if (impl_->has_native_quantised)
                    {
                        if (!rocmQuantGemm_quantizeActivationsBlockwiseScoped(
                                d_input, impl_->d_A_int8, impl_->d_scales_A_blockwise, m, k, rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] NativeQuantised blockwise activation quantization failed");
                            return false;
                        }

                        if (!rocmGemv_native_quantised_fp32_with_policy(
                                impl_->d_A_int8,
                                impl_->d_weights_native_quantised,
                                impl_->d_weights_native_scales,
                                impl_->d_weights_native_mins,
                                impl_->d_weights_native_emins,
                                d_gemv_output,
                                impl_->d_scales_A,
                                impl_->d_scatter_partial,
                                n, k,
                                impl_->native_quantised_codebook_id,
                                requireNativeQuantisedArithmeticPolicyCodebook(
                                    impl_->native_quantised_codebook_id,
                                    impl_->native_source_identity),
                                rocm_device_id_, gpu_stream_,
                                impl_->d_scales_A_blockwise))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] NativeQuantised GEMV failed");
                            return false;
                        }

                        if (beta != 0.0f)
                        {
                            // Each epilogue lane reads its own previous C value before
                            // publishing the result. Device-local C is the final destination.
                            float *const epilogue_output = gemv_output_is_mapped ? d_gemv_output : d_output;
                            if (!rocmQuantGemm_applyFp32CombineEpilogue(
                                    d_gemv_output,
                                    epilogue_output,
                                    d_output,
                                    d_bias,
                                    m,
                                    n,
                                    alpha,
                                    beta,
                                    rocm_device_id_,
                                    gpu_stream_))
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] NativeQuantised beta epilogue failed");
                                return false;
                            }
                        }
                        else if (alpha != 1.0f || d_bias)
                        {
                            if (!rocmQuantGemm_applyFp32Epilogue(
                                    d_gemv_output,
                                    d_bias,
                                    m,
                                    n,
                                    alpha,
                                    rocm_device_id_,
                                    gpu_stream_))
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] NativeQuantised epilogue failed");
                                return false;
                            }
                        }
                        // Bulk DMA is reserved for the mapped host destination.
                        if (gemv_output_is_mapped && hipMemcpyAsync(
                                d_output, d_gemv_output, static_cast<size_t>(n) * sizeof(float),
                                hipMemcpyDeviceToDevice, static_cast<hipStream_t>(gpu_stream_)) != hipSuccess)
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] NativeQuantised mapped-output publication failed");
                            return false;
                        }
                        publishROCmTensorWrite(*C, rocm_device_id_, gpu_stream_);
                        return true;
                    }

                    // The admitted VNNI program owns final device columns including beta.
                    if (d_weights_vnni)
                    {
                        if (!rocmQuantGemm_quantizeActivationsBlockwiseScoped(
                                d_input, impl_->d_A_int8, impl_->d_scales_A_blockwise, m, k, rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] INT8 blockwise activation quantization failed");
                            return false;
                        }

                        float *const int8_output = gemv_output_is_mapped ? impl_->d_C_fp32 : d_output;
                        const float *d_existing = (beta != 0.0f) ? d_output : nullptr;
                        const bool int8_decode_ok = rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                            impl_->d_A_int8, d_weights_vnni, int8_output,
                            impl_->d_scales_A_blockwise, d_scales_B,
                            n, k,
                            alpha, beta,
                            d_existing, d_bias,
                            rocm_device_id_, gpu_stream_);

                        if (!int8_decode_ok)
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] INT8 blockwise GEMV failed");
                            return false;
                        }

                        if (gemv_output_is_mapped && hipMemcpyAsync(d_output, int8_output,
                                static_cast<size_t>(n) * sizeof(float), hipMemcpyDeviceToDevice,
                                static_cast<hipStream_t>(gpu_stream_)) != hipSuccess)
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] INT8 mapped-output publication failed");
                            return false;
                        }
                        publishROCmTensorWrite(*C, rocm_device_id_, gpu_stream_);
                        return true;
                    }

                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] M=1 decode has no supported GEMV path for alpha="
                              << alpha << " beta=" << beta
                              << " native_quantised=" << (impl_->has_native_quantised ? "true" : "false")
                              << " int8_vnni=" << (d_weights_vnni ? "true" : "false"));
                    return false;
                }
                // No VNNI weight pointers available for M=1 decode — this should
                // not happen unless weights failed to upload.
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] M=1 decode: "
                          "no native-quantised or INT8-VNNI weights available");
                return false;
            }

            // =========================================================================
            // NATIVE QUANTISED PREFILL ROUTE (M>1)
            //
            // We defer actual native dispatch until after activation quantization,
            // because both native prefill formats consume INT8 activations.
            // =========================================================================

            // Ensure weights are uploaded to device
            if (phase_timing)
                phase_start = std::chrono::high_resolution_clock::now();
            ensureWeightsConverted();
            if (currentProjectionRowWork(m) && (!impl_ || !impl_->has_native_quantised))
                throw std::invalid_argument("Decode projection work requires the prepared all-codebook NativeQuantised route");
            if (phase_timing)
            {
                phase_end = std::chrono::high_resolution_clock::now();
                double weights_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                if (weights_ms > 1.0)
                    LOG_TRACE("[ROCmGEMM::PHASES] ensureWeightsConverted: " << weights_ms << "ms");
            }

            // Get weight device pointers
            // Option B: weights are stored as VNNI only. For CK GEMM (needs row-major),
            // we repack VNNI→row-major into a shared workspace scratch buffer.
            float *d_scales_B = nullptr;

            if (packed_)
            {
                d_scales_B = packed_->d_scales;
            }
            else if (impl_)
            {
                d_scales_B = impl_->d_scales_B;
            }

            // NativeQuantised path uses impl_->d_weights_native_scales (not d_scales_B),
            // so d_scales_B may be null when only native-quantised weights are present
            // (e.g. weights loaded via GPU pipeline / MoE batch constructor).
            const bool native_quantised_available_mt = impl_ && impl_->has_native_quantised;
            if (!d_scales_B && !native_quantised_available_mt)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Weight scales not uploaded to device");
                return false;
            }

            if (d_scales_B && !validatePointerDeviceOrLog(
                                  d_scales_B, rocm_device_id_, "d_scales_B", "ROCmQuantisedGemmKernel::multiply_tensor"))
            {
                return false;
            }

            // Validate and populate workspace pointers (includes d_B_rowmajor_scratch)
            if (phase_timing)
                phase_start = std::chrono::high_resolution_clock::now();
            validateWorkspace();
            if (phase_timing)
            {
                phase_end = std::chrono::high_resolution_clock::now();
                double workbuf_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                if (workbuf_ms > 1.0)
                    LOG_TRACE("[ROCmGEMM::PHASES] validateWorkspace: " << workbuf_ms << "ms");
            }

            // Option B: Repack VNNI→row-major was used only by the CK fallback.
            // CK is retired; the M>1 path dispatches native-quantised / INT8-VNNI
            // directly against `impl_->d_weights_int8_vnni` and never needs a
            // row-major copy.

            int8_t *d_A_int8 = impl_->d_A_int8;
            float *d_scales_A = impl_->d_scales_A;
            int32_t *d_C_int32 = impl_->d_C_int32;

            if (!validatePointerDeviceOrLog(
                    d_A_int8, rocm_device_id_, "workspace::QUANT_A", "ROCmQuantisedGemmKernel::multiply_tensor") ||
                !validatePointerDeviceOrLog(
                    d_scales_A, rocm_device_id_, "workspace::SCALES_A", "ROCmQuantisedGemmKernel::multiply_tensor") ||
                !validatePointerDeviceOrLog(
                    d_C_int32, rocm_device_id_, "workspace::ACC_INT32", "ROCmQuantisedGemmKernel::multiply_tensor"))
            {
                return false;
            }

            LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Work buffers: A_int8=" << (void *)d_A_int8
                                                                                         << " scales_A=" << (void *)d_scales_A << " C_int32=" << (void *)d_C_int32);

            float *d_A_fp32_src = const_cast<float *>(d_input);
            LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Using GPU input directly: " << d_A_fp32_src);

            LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Now quantizing activations");

            // Quantize activations FP32 → INT8 (blockwise)
            // M=1 decode uses the GEMV path above with its own blockwise quantization.
            if (phase_timing)
                phase_start = std::chrono::high_resolution_clock::now();
            {
                const int act_block_k = rocmGemv_int8_vnni_get_act_block_k();
                if (!rocmQuantGemm_quantizeActivationsBlockwiseScoped(d_A_fp32_src, d_A_int8, impl_->d_scales_A_blockwise, m, k, rocm_device_id_, gpu_stream_, act_block_k))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Failed to blockwise-quantize activations");
                    return false;
                }
            }
            if (phase_timing)
            {
                phase_end = std::chrono::high_resolution_clock::now();
                double quant_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                if (quant_ms > 1.0)
                    LOG_TRACE("[ROCmGEMM::PHASES] quantizeActivations: " << quant_ms << "ms");
            }

            // Attempt native dual-format prefill path first (INT8 VNNI or ratio-VNNI).
            // If this returns false, we continue to the CK path below.
            if (m > 1)
            {
                // CRITICAL OPTIMIZATION: Detect mapped (host-pinned) output memory.
                // Mapped memory is used for logits (CPU sampling needs host-side access).
                // GPU writes to mapped memory go over PCIe (~1.6 GB/s) instead of HBM
                // (~1024 GB/s). For LM_Head (M=596, N=152064 → 362MB), this causes
                // 226ms of PCIe writes vs. 0.35ms HBM + 30ms streaming D2H = 30ms.
                //
                // Fix: redirect scaling output to device workspace (d_C_fp32), then
                // do a bulk hipMemcpyAsync to the mapped output.
                const bool output_is_mapped = C_fp32->isMapped();
                const bool output_needs_copyout = output_is_mapped;
                float *d_prefill_output = output_needs_copyout ? impl_->d_C_fp32 : d_output;

                // One-time diagnostic for mapped output redirect
                if (output_is_mapped && n > 100000)
                {
                    static std::once_flag mapped_redirect_once;
                    std::call_once(mapped_redirect_once, [&]()
                                   { LOG_WARN("[multiply_tensor] MAPPED OUTPUT REDIRECT: LM_Head M=" << m << " N=" << n
                                                                                                     << " gpu_ptr=" << d_output
                                                                                                     << " -> d_C_fp32=" << impl_->d_C_fp32
                                                                                                     << " (" << (static_cast<size_t>(m) * n * 4 / (1024 * 1024)) << " MB)"); });
                }

                const float *d_prefill_bias = nullptr;
                if (bias)
                {
                    auto *bias_tensor = const_cast<TensorBase *>(bias);
                    const auto target_device =
                        DeviceId::rocm(rocm_device_id_);
                    TransferEngine::requireDeviceInput(
                        bias_tensor, target_device, gpu_stream_);

                    d_prefill_bias = static_cast<const float *>(bias_tensor->gpu_data_ptr());
                }

                // Try native-quantised GEMM first (halved HBM bandwidth for Q4_0/IQ4_NL)
                // This path reads compact native-quantised payloads (4.5 bpw) directly,
                // decodes to INT8 in-register, and produces FP32 output with per-block
                // FP16 scales applied inline — no separate scaling epilogue needed.
                // Bias is applied via a lightweight biasAdd kernel after the GEMM.
                if (impl_->has_native_quantised && alpha == 1.0f && beta == 0.0f)
                {
                    const uint8_t cb_id = impl_->native_quantised_codebook_id;
                    {
                        if (rocmGemm_native_quantised_fp32_with_policy(
                                d_A_int8,
                                impl_->d_weights_native_quantised,
                                impl_->d_weights_native_scales,
                                impl_->d_weights_native_mins,
                                impl_->d_weights_native_emins,
                                d_prefill_output,
                                d_scales_A,
                                impl_->d_scales_A_blockwise,
                                m, n, k,
                                cb_id,
                                requireNativeQuantisedArithmeticPolicyCodebook(
                                    cb_id,
                                    impl_->native_source_identity),
                                rocm_device_id_, gpu_stream_))
                        {
                            LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] "
                                      "NativeQuantised GEMM succeeded (M="
                                      << m << " N=" << n << " K=" << k
                                      << " codebook=" << static_cast<int>(cb_id)
                                      << (d_prefill_bias ? " +bias" : "") << ")");

                            // Apply bias if present (same post-kernel pattern as GEMV)
                            if (d_prefill_bias)
                            {
                                if (!rocmQuantGemm_biasAdd(d_prefill_output, d_prefill_bias, m, n, rocm_device_id_, gpu_stream_))
                                {
                                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] NativeQuantised GEMM bias add failed");
                                    return false;
                                }
                            }

                            // Copy from workspace to output if needed.
                            if (output_needs_copyout)
                            {
                                const size_t native_quantised_c_size = static_cast<size_t>(m) * n;
                                const hipError_t copy_err = hipMemcpyAsync(
                                    d_output,
                                    d_prefill_output,
                                    native_quantised_c_size * sizeof(float),
                                    hipMemcpyDeviceToDevice,
                                    static_cast<hipStream_t>(gpu_stream_));
                                if (copy_err != hipSuccess)
                                {
                                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Device-alias output publication failed: "
                                              << hipGetErrorString(copy_err));
                                    return false;
                                }
                            }
                            publishROCmTensorWrite(*C, rocm_device_id_, gpu_stream_);
                            return true;
                        }
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] "
                                  "NativeQuantised GEMM failed; refusing INT8 GEMM fallback");
                        return false;
                    }
                }

                if (tryPrefillNativeGemm(
                        d_A_int8,
                        d_prefill_output,
                        d_scales_A,
                        impl_->d_scales_A_blockwise,
                        d_scales_B,
                        d_prefill_bias,
                        m, n, k,
                        alpha, beta,
                        "ROCmQuantisedGemmKernel::multiply_tensor"))
                {
                    // Copy from device workspace to final destination when scaling
                    // was redirected away from d_output.
                    if (output_is_mapped)
                    {
                        const size_t prefill_c_size = static_cast<size_t>(m) * n;
                        // A mapped tensor exposes a device alias. Publish to
                        // that alias on the executor stream and leave host
                        // visibility to the explicit result boundary.
                        const hipError_t copy_err = hipMemcpyAsync(
                            d_output,
                            impl_->d_C_fp32,
                            prefill_c_size * sizeof(float),
                            hipMemcpyDeviceToDevice,
                            static_cast<hipStream_t>(gpu_stream_));
                        if (copy_err != hipSuccess)
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Device-alias output publication failed: "
                                      << hipGetErrorString(copy_err));
                            return false;
                        }
                    }
                    publishROCmTensorWrite(*C, rocm_device_id_, gpu_stream_);
                    return true;
                }

                // Both VNNI paths failed for M>1 prefill — hard error.
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] M>1 prefill: "
                          "both native-quantised and INT8-VNNI paths failed (M="
                          << m << " N=" << n << " K=" << k << ").");
                return false;
            }

            // Unreachable: M>1 path always returns above via native/INT8-VNNI or hard error.
            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Reached unreachable CK fallback sentinel");
            return false;
        }

        bool ROCmQuantisedGemmKernel::multiply_tensor_timed(
            const TensorBase * /*A*/, TensorBase * /*C*/,
            int /*m*/, int /*n*/, int /*k*/,
            float *kernel_time_ms)
        {
            // CK timed path retired. The timed variant existed solely to measure
            // CK kernel execution time for benchmarking. Use multiply_tensor() for
            // functional execution via native-quantised / INT8-VNNI paths.
            if (kernel_time_ms)
                *kernel_time_ms = -1.0f;
            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_timed] CK timed path retired");
            return false;
        }

        bool ROCmQuantisedGemmKernel::multiply_fused_tensor(
            const TensorBase *input,
            const std::vector<TensorProjectionDesc> &projections,
            int m, int k,
            const IMPIContext *mpi_ctx,
            DeviceWorkspaceManager *workspace)
        {
            // Decode work admission never changes the selected arithmetic program.
            if (m > 1 && currentProjectionRowWork(m) && !VerifierKernelModeScope::active())
                throw std::invalid_argument("Projection row intervals require explicit grouped Decode arithmetic for M>1");

            for (const auto &projection : projections)
            {
                projection.requireOriginalOutputSlice(k);
                auto original_scope = projection.beginOriginalOutputSliceScope();
            }

            auto execution_stream =
                static_cast<hipStream_t>(requireGPUStream());
            const DeviceId target_device = DeviceId::rocm(rocm_device_id_);
            ROCM_KERNEL_PROFILE_SCOPE_STREAM(
                ROCmKernelType::GEMM, execution_stream);

            // Use passed workspace if provided, otherwise fall back to bound workspace
            DeviceWorkspaceManager *ws = workspace ? workspace : workspace_;
            DeviceWorkspaceManager *saved_workspace = workspace_;
            if (ws && ws != workspace_)
            {
                // Temporarily bind passed workspace for this call
                workspace_ = ws;
            }
            auto restore_workspace = [&](void *)
            {
                if (ws && ws != saved_workspace)
                {
                    workspace_ = saved_workspace;
                }
            };
            std::unique_ptr<void, decltype(restore_workspace)> workspace_restore_guard(
                (ws && ws != saved_workspace) ? static_cast<void *>(this) : nullptr,
                restore_workspace);

            if (!input || projections.empty())
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Null input or empty projections");
                return false;
            }

            if (m <= 0 || k <= 0)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Invalid dimensions: m=" << m << " k=" << k);
                return false;
            }

            /**
             * Publish every fused projection through TensorBase's event-backed
             * device-write transition.
             *
             * Fused GEMM launches write directly into caller-owned output tensors.
             * Returning while those tensors remain SYNCED leaves their stale host
             * bytes apparently valid and also gives a later consumer stream no
             * producer event to wait on. Keep publication centralized so every
             * optimized return path has the same ownership semantics.
             */
            const auto publish_projection_outputs = [&]()
            {
                for (const TensorProjectionDesc &projection : projections)
                {
                    auto *output = dynamic_cast<TensorBase *>(projection.output);
                    if (!output)
                    {
                        throw std::logic_error(
                            "ROCm fused GEMM output publication requires event-capable TensorBase storage");
                    }
                    publishROCmTensorWrite(
                        *output,
                        rocm_device_id_,
                        gpu_stream_);
                }
            };

            rocmQuantGemm_setDevice(rocm_device_id_);

            // Step 1: Ensure input is on the GPU
            const float *d_input = nullptr;
            if (input->native_type() == TensorType::FP32)
            {
                auto *fp32_input = dynamic_cast<FP32Tensor *>(const_cast<TensorBase *>(input));
                if (!fp32_input)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Failed to cast input to FP32Tensor");
                    return false;
                }
                TransferEngine::requireDeviceInput(
                    fp32_input,
                    DeviceId::rocm(rocm_device_id_),
                    gpu_stream_);
                d_input = static_cast<const float *>(fp32_input->gpu_data_ptr());
                // NOTE: Don't log fp32_input->data() here - it triggers D2H transfer!
                LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Input GPU ptr=" << d_input);
            }
            else
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Unsupported input type: "
                          << static_cast<int>(input->native_type()));
                return false;
            }

            if (!d_input)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Input has no GPU data");
                return false;
            }

            // Step 2: Validate and populate workspace pointers (shared across all projections)
            validateWorkspace();

            // Step 3: Ensure all projection weights are on GPU before the native-quantised
            // pre-scan. has_native_quantised is only set by ensureWeightsConverted(), so we
            // must call it on every projection kernel before checking the flag.
            for (size_t pi = 0; pi < projections.size(); ++pi)
            {
                auto *rk = dynamic_cast<ROCmQuantisedGemmKernel *>(projections[pi].kernel);
                if (rk)
                {
                    rk->ensureWeightsConverted();
                    // Reject foreign original shapes before even shared activation
                    // quantization can read an invalid K extent.
                    if(m==1 && projections.size()>=2 && rk->impl_ && rk->impl_->has_native_quantised &&
                        (rk->rocm_device_id_!=rocm_device_id_ || rk->N_!=projections[pi].n || rk->K_!=k))
                        throw std::invalid_argument("Fused M1 original prepared device/N/K differs from its descriptor");
                }
            }

            // Step 4: Quantize activations ONCE (shared across all projections)
            // All M=1 paths now use INT8 scatter kernels which require pre-quantized
            // activations. For M>1, CK GEMM also uses pre-quantized activations.
            //
            // Blockwise quantization is supported when ALL projection kernels support it.
            // - NativeQuantised (Q4_0/IQ4_NL): blockwise at any M.
            // - INT8-VNNI (Q8_0/Q8_1): blockwise for both decode and prefill.
            bool all_projections_native_quantised = true;
            bool all_projections_have_vnni_weights = true;
            for (size_t pi = 0; pi < projections.size(); ++pi)
            {
                auto *rk = dynamic_cast<ROCmQuantisedGemmKernel *>(projections[pi].kernel);
                if (!rk || !rk->impl_ || !rk->impl_->has_native_quantised)
                {
                    all_projections_native_quantised = false;
                }
                if (!rk || !rk->impl_ || (!rk->impl_->has_native_quantised && !rk->impl_->d_weights_int8_vnni))
                {
                    all_projections_have_vnni_weights = false;
                }
                if (!all_projections_native_quantised && !all_projections_have_vnni_weights)
                    break;
            }
            bool fused_uses_blockwise_shared_quant = false;
            {
                LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Quantizing activations once, m=" << m << " k=" << k);

                const bool needs_block_sums =
                    g_rocm_native_quantised_decode_equivalent_scope &&
                    all_projections_native_quantised && (m >= 2 || countedVerifierRows(m)) && (k % 32) == 0;
                const bool quant_ok = needs_block_sums
                    ? rocmQuantGemm_quantizeActivationsBlockwiseWithSumsScoped(
                          const_cast<float *>(d_input),
                          impl_->d_A_int8,
                          impl_->d_scales_A_blockwise,
                          impl_->d_sums_A_blockwise,
                          m, k, rocm_device_id_, gpu_stream_)
                    : rocmQuantGemm_quantizeActivationsBlockwiseScoped(
                          const_cast<float *>(d_input),
                          impl_->d_A_int8,
                          impl_->d_scales_A_blockwise,
                          m, k, rocm_device_id_, gpu_stream_);
                if (!quant_ok)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Blockwise activation quantization failed");
                    return false;
                }

                fused_uses_blockwise_shared_quant = all_projections_have_vnni_weights;

        if (m >= 2 && projections.size() >= 2 &&
            PerfStatsCollector::isDomainEnabled("kernel"))
                {
                    PerfStatsCollector::addCounter(
                        "kernel",
                        "rocm_fused_small_m_shared_quant_calls",
                        1.0,
                        "gemm",
                        "rocm:" + std::to_string(rocm_device_id_),
                        PerfStatsCollector::Tags{
                            {"m", std::to_string(m)},
                            {"k", std::to_string(k)},
                            {"projections", std::to_string(projections.size())}});

                    if (m == 2)
                    {
                        PerfStatsCollector::addCounter(
                            "kernel",
                            "rocm_fused_m2_shared_quant_calls",
                            1.0,
                            "gemm",
                            "rocm:" + std::to_string(rocm_device_id_),
                            PerfStatsCollector::Tags{
                                {"k", std::to_string(k)},
                                {"projections", std::to_string(projections.size())}});
                    }
                }
            }

            // Step 5: Execute projections using the SHARED quantized activations
            // For M=1, VNNI projections are batched into a single kernel dispatch.
            // NativeQuantised (Q4/IQ4) and M>1 projections are dispatched individually.
            bool all_success = true;

            // Batched INT8 scatter GEMV collection arrays (for M=1 VNNI projections)
            const int8_t *batch_B_ptrs[8] = {};
            float *batch_C_ptrs[8] = {};
            const float *batch_scales_B_ptrs[8] = {};
            const float *batch_bias_ptrs[8] = {};
            int batch_N[8] = {};
            int batch_count = 0;

            auto isGDNDecodeProjectionSet = [&]() -> bool
            {
                if (m != 1 || projections.size() != 4)
                    return false;

                constexpr std::array<const char *, 4> expected_names = {"qkv", "z", "alpha", "beta"};
                for (size_t i = 0; i < expected_names.size(); ++i)
                {
                    const char *name = projections[i].name;
                    if (!name || std::strcmp(name, expected_names[i]) != 0)
                        return false;
                }
                return true;
            };

            /**
             * Join every optional bias producer to the transaction root before
             * the projection streams fork from quant_ready.  TP biases are
             * TensorSlice wrappers around their presliced FP32 storage; treating
             * the wrapper itself as FP32 silently drops the learned bias.  The
             * root-stream join also makes an asynchronous startup publication an
             * explicit dependency in eager execution and native graph capture.
             */
            const auto join_projection_biases_to_root =
                [&](const char *callsite)
            {
                for (size_t pi = 0; pi < projections.size(); ++pi)
                {
                    const auto &projection = projections[pi];
                    if (!projection.bias)
                        continue;

                    const TensorBase *bias_tensor = projection.bias;
                    if (auto *slice =
                            dynamic_cast<const TensorSlice *>(projection.bias))
                    {
                        bias_tensor = slice->inner();
                    }

                    auto *fp32_bias = dynamic_cast<FP32Tensor *>(
                        const_cast<TensorBase *>(bias_tensor));
                    if (!fp32_bias)
                    {
                        throw std::runtime_error(
                            std::string("[") + callsite + "] Projection " +
                            std::to_string(pi) + " bias is not FP32Tensor");
                    }

                    const auto current_device = fp32_bias->current_device();
                    if (current_device.has_value() &&
                        current_device->is_gpu() &&
                        current_device.value() != target_device)
                    {
                        throw std::runtime_error(
                            std::string("[") + callsite + "] Projection " +
                            std::to_string(pi) + " bias is resident on " +
                            current_device->to_string() + " instead of " +
                            target_device.toString());
                    }

                    TransferEngine::requireDeviceInput(
                        fp32_bias,
                        target_device,
                        execution_stream);
                    if (!fp32_bias->gpu_data_ptr())
                    {
                        throw std::runtime_error(
                            std::string("[") + callsite + "] Projection " +
                            std::to_string(pi) +
                            " bias has no device storage after root-stream join");
                    }
                }
            };

            // =========================================================================
            // CONCURRENT PREFILL PATH: Multi-stream dispatch for M>1 projections
            //
            // When enabled (LLAMINAR_ROCM_CONCURRENT_PREFILL=1), overlaps fused GEMM
            // projections on separate HIP streams.  Each projection gets its own
            // stream + scratch buffer so blockwise V7 kernels can write intermediate
            // results without conflicting.
            //
            // Prerequisites:
            //   - M > 1 (prefill, not decode)
            //   - >= 2 projections
            //   - All projections have VNNI weights (tryPrefillNativeGemm path)
            //   - No mapped outputs (those need shared d_C_fp32)
            //
            // DAG: quantization (main stream)
            //        ├──> projection 0 (stream 0, scratch 0)
            //        ├──> projection 1 (stream 1, scratch 1)
            //        └──> ...
            //      main stream waits on all completion events
            // =========================================================================
#ifdef HAVE_ROCM
            const bool small_m_native_verifier_fused =
                g_rocm_native_quantised_decode_equivalent_scope &&
                all_projections_native_quantised && (m >= 2 || countedVerifierRows(m));
            if (m > 2 && projections.size() >= 2 &&
                fused_uses_blockwise_shared_quant &&
                all_projections_native_quantised &&
                !small_m_native_verifier_fused &&
                debugEnv().rocm.concurrent_prefill)
            {
                // Check eligibility: all outputs must be direct device memory (not BAR/mapped)
                bool concurrent_eligible = true;
                for (size_t i = 0; i < projections.size(); ++i)
                {
                    const auto &proj = projections[i];
                    auto original_scope = proj.beginOriginalOutputSliceScope();
                    if (!proj.kernel || !proj.output)
                    {
                        concurrent_eligible = false;
                        break;
                    }
                    auto *fp32_out = dynamic_cast<FP32Tensor *>(proj.output);
                    if (!fp32_out)
                    {
                        concurrent_eligible = false;
                        break;
                    }
                    if (fp32_out->isMapped())
                    {
                        concurrent_eligible = false;
                        break;
                    }
                    auto *rk = dynamic_cast<ROCmQuantisedGemmKernel *>(proj.kernel);
                    if (!rk || !rk->impl_)
                    {
                        concurrent_eligible = false;
                        break;
                    }
                }

                if (concurrent_eligible)
                {
                    const int num_proj = static_cast<int>(projections.size());
                    auto submission = acquireHIPProjectionSubmission(rocm_device_id_, gpu_stream_);
                    const auto pool = hipProjectionView(submission.resources());

                    join_projection_biases_to_root("ConcurrentPrefill");

                    // Record event after quantization completes on main stream
                    (void)hipEventRecord(pool.quant_ready,
                                   static_cast<hipStream_t>(gpu_stream_));

                    for (int pi = 0; pi < num_proj; ++pi)
                    {
                        const auto &proj = projections[pi];
                        auto original_scope = proj.beginOriginalOutputSliceScope();
                        auto *rocm_kernel = dynamic_cast<ROCmQuantisedGemmKernel *>(proj.kernel);
                        const int n = proj.n;

                        // Validate and prepare this projection's workspace
                        rocm_kernel->validateWorkspace();

                        // Get output pointer
                        auto *fp32_output = dynamic_cast<FP32Tensor *>(proj.output);
                        float *d_output = static_cast<float *>(fp32_output->gpu_data_ptr());

                        if (!d_output)
                        {
                            throw std::runtime_error(
                                "[ConcurrentPrefill] Projection " + std::to_string(pi) +
                                " output has no GPU data (shape=" +
                                std::to_string(fp32_output->rows()) + "x" +
                                std::to_string(fp32_output->cols()) +
                                ", coherence=" +
                                std::to_string(static_cast<int>(fp32_output->coherenceState())) +
                                ") — ensure scratch tensors call allocateOnDevice()");
                        }

                        // Get per-projection weight scales and bias
                        const float *d_scales_B = nullptr;
                        if (rocm_kernel->packed_)
                        {
                            d_scales_B = rocm_kernel->packed_->d_scales;
                        }
                        else if (rocm_kernel->impl_)
                        {
                            d_scales_B = rocm_kernel->impl_->d_scales_B;
                        }
                        if (!d_scales_B && !(rocm_kernel->impl_ && rocm_kernel->impl_->has_native_quantised))
                        {
                            LOG_WARN("[ConcurrentPrefill] Projection " << pi
                                                                       << " (" << (proj.name ? proj.name : "?")
                                                                       << ") d_scales_B=0 — packed_=" << (const void *)rocm_kernel->packed_
                                                                       << " packed_->d_scales=" << (rocm_kernel->packed_ ? (const void *)rocm_kernel->packed_->d_scales : nullptr)
                                                                       << " impl_=" << (const void *)rocm_kernel->impl_.get()
                                                                       << " impl_->d_scales_B=" << (rocm_kernel->impl_ ? (const void *)rocm_kernel->impl_->d_scales_B : nullptr)
                                                                       << " weights_converted=" << rocm_kernel->weights_converted_
                                                                       << " N=" << rocm_kernel->N_ << " K=" << rocm_kernel->K_
                                                                       << " device=" << rocm_kernel->rocm_device_id_);
                        }
                        const float *d_bias = nullptr;
                        if (proj.bias)
                        {
                            const TensorBase *bias_tensor = proj.bias;
                            if (auto *slice =
                                    dynamic_cast<const TensorSlice *>(proj.bias))
                            {
                                bias_tensor = slice->inner();
                            }
                            auto *bias_fp32 = dynamic_cast<FP32Tensor *>(
                                const_cast<TensorBase *>(bias_tensor));
                            if (!bias_fp32)
                            {
                                throw std::runtime_error(
                                    "[ConcurrentPrefill] Projection " +
                                    std::to_string(pi) +
                                    " bias is not FP32Tensor");
                            }
                            d_bias = static_cast<const float *>(
                                bias_fp32->gpu_data_ptr());
                            if (!d_bias)
                            {
                                throw std::runtime_error(
                                    "[ConcurrentPrefill] Projection " +
                                    std::to_string(pi) +
                                    " bias lost its prejoined device storage");
                            }
                        }

                        int stream_idx = pi % pool.count;

                        // This stream waits for quantization to complete
                        (void)hipStreamWaitEvent(pool.streams[stream_idx], pool.quant_ready, 0);

                        // If projections > streams, wait for previous projection on this stream
                        if (pi >= pool.count)
                        {
                            (void)hipStreamWaitEvent(pool.streams[stream_idx],
                                               pool.completion[stream_idx], 0);
                        }

                        // Dispatch with stream + scratch overrides
                        LOG_TRACE("[ConcurrentPrefill] Projection " << pi
                                                                    << " (" << (proj.name ? proj.name : "?")
                                                                    << ") M=" << m << " N=" << n << " K=" << k
                                                                    << " on stream " << stream_idx);

                        bool proj_ok = rocm_kernel->tryPrefillNativeGemm(
                            impl_->d_A_int8,
                            d_output,
                            impl_->d_scales_A,
                            impl_->d_scales_A_blockwise,
                            d_scales_B,
                            d_bias,
                            m, n, k,
                            1.0f, 0.0f,
                            "ConcurrentPrefill",
                            pool.streams[stream_idx],
                            nullptr);

                        if (!proj_ok)
                        {
                            throw std::runtime_error(
                                "[ConcurrentPrefill] Projection " + std::to_string(pi) +
                                " (" + std::string(proj.name ? proj.name : "?") +
                                ") kernel launch failed on stream " + std::to_string(stream_idx) +
                                " — cannot continue inference");
                        }

                        // Record completion event for this stream
                        (void)hipEventRecord(pool.completion[stream_idx],
                                       pool.streams[stream_idx]);
                    }

                    // All projections dispatched — main stream waits for completion
                    for (int si = 0; si < std::min(num_proj, pool.count); ++si)
                    {
                        (void)hipStreamWaitEvent(
                            static_cast<hipStream_t>(gpu_stream_),
                            pool.completion[si], 0);
                    }

                    /* Prove the persistent side-stream prefill branch ran. */
                if (PerfStatsCollector::isDomainEnabled("kernel"))
                    {
                        PerfStatsCollector::addCounter(
                            "kernel",
                            "rocm_fused_projection_stream_pool_calls",
                            1.0,
                            "gemm",
                            "rocm:" + std::to_string(rocm_device_id_),
                            PerfStatsCollector::Tags{
                                {"mode", "prefill"},
                                {"m", std::to_string(m)},
                                {"k", std::to_string(k)},
                                {"projections", std::to_string(num_proj)},
                                {"streams", std::to_string(
                                     std::min(num_proj, pool.count))}});
                    }

                    LOG_TRACE("[ConcurrentPrefill] All " << num_proj
                                                         << " projections dispatched concurrently");

                    // Restore workspace and return success
                    if (ws && ws != saved_workspace)
                        workspace_ = saved_workspace;
                    publish_projection_outputs();
                    return true;
                }
            }
#endif // HAVE_ROCM

            // =========================================================================
            // CONCURRENT DECODE PATH: Multi-stream dispatch for M=1 GEMV projections
            //
            // When enabled globally (LLAMINAR_ROCM_CONCURRENT_DECODE=1) or for a
            // scoped projection set (GDN), overlaps fused GEMV
            // projections on separate HIP streams.  At small TP-sharded N dimensions,
            // individual GEMVs may not saturate all CUs, so overlapping them can
            // improve utilization.
            //
            // INT8-VNNI projections use rocmGemv_int8_int8_fp32_vnni_blockwise_scaled
            //   which has no shared scratch dependency (atomicAdd to output).
            // NativeQuantised projections use rocmGemv_native_quantised_fp32 which needs a
            //   per-stream scatter_partial buffer from the pool.
            //
            // DAG: quantization (main stream)
            //        ├──> projection 0 GEMV (stream 0)
            //        ├──> projection 1 GEMV (stream 1)
            //        └──> ...
            //      main stream waits on all completion events
            // =========================================================================
#ifdef HAVE_ROCM
            const bool use_concurrent_decode =
                debugEnv().rocm.concurrent_decode ||
                (debugEnv().rocm.gdn_concurrent_decode && isGDNDecodeProjectionSet());

            if (m == 1 && projections.size() >= 2 &&
                fused_uses_blockwise_shared_quant &&
                use_concurrent_decode)
            {
                bool decode_concurrent_eligible = true;
                for (size_t i = 0; i < projections.size(); ++i)
                {
                    const auto &proj = projections[i];
                    auto original_scope = proj.beginOriginalOutputSliceScope();
                    if (!proj.kernel || !proj.output)
                    {
                        decode_concurrent_eligible = false;
                        break;
                    }
                    auto *fp32_out = dynamic_cast<FP32Tensor *>(proj.output);
                    if (!fp32_out)
                    {
                        decode_concurrent_eligible = false;
                        break;
                    }
                    // Mapped outputs use shared d_C_fp32 — cannot overlap
                    if (fp32_out->isMapped())
                    {
                        decode_concurrent_eligible = false;
                        break;
                    }
                    auto *rk = dynamic_cast<ROCmQuantisedGemmKernel *>(proj.kernel);
                    if (!rk || !rk->impl_)
                    {
                        decode_concurrent_eligible = false;
                        break;
                    }
                }

                if (decode_concurrent_eligible)
                {
                    const int num_proj = static_cast<int>(projections.size());
                    auto submission = acquireHIPProjectionSubmission(rocm_device_id_, gpu_stream_);
                    const auto pool = hipProjectionView(submission.resources());
                    join_projection_biases_to_root("ConcurrentDecode");
                    const std::string batched_partial_name =
                        scatterPartialBatchedBufferName();
                    const size_t batched_partial_bytes =
                        workspace_->getBufferSize(batched_partial_name);
                    if (!impl_->d_scatter_partial_batched ||
                        batched_partial_bytes == 0)
                    {
                        throw std::runtime_error(
                            "[ConcurrentDecode] Missing graph-planned batched "
                            "scatter workspace");
                    }

                    /*
                     * Match appendFusedProjectionWorkspaceRequirements(): a
                     * stream reused by later projections owns one slice sized
                     * to the widest projection assigned to it.  Prefix offsets
                     * avoid the old equal-quarter layout, which forced a narrow
                     * Q/K/V or GDN bundle to reserve four copies of its widest
                     * member.
                     */
                    std::array<size_t,
                               ROCM_NATIVE_SMALL_M_WORKSPACE_BATCH_PROJECTIONS>
                        scatter_slot_columns{};
                    for (int pi = 0; pi < num_proj; ++pi)
                    {
                        const int width = projections[static_cast<size_t>(pi)].n;
                        if (width <= 0)
                        {
                            throw std::runtime_error(
                                "[ConcurrentDecode] Fused projection has a non-positive width");
                        }
                        const size_t stream =
                            static_cast<size_t>(pi % pool.count);
                        scatter_slot_columns[stream] = std::max(
                            scatter_slot_columns[stream],
                            static_cast<size_t>(width));
                    }
                    std::array<size_t,
                               ROCM_NATIVE_SMALL_M_WORKSPACE_BATCH_PROJECTIONS>
                        scatter_slot_offsets{};
                    size_t required_batched_bytes = 0;
                    for (int stream = 0; stream < pool.count; ++stream)
                    {
                        scatter_slot_offsets[static_cast<size_t>(stream)] =
                            required_batched_bytes;
                        required_batched_bytes +=
                            static_cast<size_t>(ROCM_NATIVE_SMALL_M_GRAPH_SAFE_KB_CAP) *
                            scatter_slot_columns[static_cast<size_t>(stream)] *
                            sizeof(float);
                    }
                    if (required_batched_bytes > batched_partial_bytes)
                    {
                        throw std::runtime_error(
                            "[ConcurrentDecode] Graph-planned batched scatter "
                            "workspace is smaller than the fused projection layout");
                    }

                    // Record event after quantization completes on main stream
                    (void)hipEventRecord(pool.quant_ready,
                                   static_cast<hipStream_t>(gpu_stream_));

                    for (int pi = 0; pi < num_proj; ++pi)
                    {
                        const auto &proj = projections[pi];
                        auto original_scope = proj.beginOriginalOutputSliceScope();
                        auto *rocm_kernel = dynamic_cast<ROCmQuantisedGemmKernel *>(proj.kernel);
                        const int n = proj.n;

                        auto *fp32_output = dynamic_cast<FP32Tensor *>(proj.output);
                        float *d_output = static_cast<float *>(fp32_output->gpu_data_ptr());

                        if (!d_output)
                        {
                            throw std::runtime_error(
                                "[ConcurrentDecode] Projection " + std::to_string(pi) +
                                " output has no GPU data — cannot continue inference");
                        }

                        const float *d_scales_B = rocm_kernel->impl_->d_scales_B;
                        const float *d_bias = nullptr;
                        if (proj.bias)
                        {
                            const TensorBase *bias_tensor = proj.bias;
                            if (auto *slice =
                                    dynamic_cast<const TensorSlice *>(proj.bias))
                            {
                                bias_tensor = slice->inner();
                            }
                            auto *bias_fp32 = dynamic_cast<FP32Tensor *>(
                                const_cast<TensorBase *>(bias_tensor));
                            if (!bias_fp32)
                            {
                                throw std::runtime_error(
                                    "[ConcurrentDecode] Projection " +
                                    std::to_string(pi) +
                                    " bias is not FP32Tensor");
                            }
                            d_bias = static_cast<const float *>(
                                bias_fp32->gpu_data_ptr());
                            if (!d_bias)
                            {
                                throw std::runtime_error(
                                    "[ConcurrentDecode] Projection " +
                                    std::to_string(pi) +
                                    " bias lost its prejoined device storage");
                            }
                        }

                        int stream_idx = pi % pool.count;

                        // This stream waits for quantization
                        (void)hipStreamWaitEvent(pool.streams[stream_idx], pool.quant_ready, 0);

                        // If reusing a stream, wait for its previous work
                        if (pi >= pool.count)
                        {
                            (void)hipStreamWaitEvent(pool.streams[stream_idx],
                                               pool.completion[stream_idx], 0);
                        }

                        bool proj_ok = false;

                        if (rocm_kernel->impl_->has_native_quantised)
                        {
                            // Each stream receives its exact disjoint prefix
                            // slice; a reused stream retains the widest width
                            // assigned to that stream.
                            const size_t required_partial_bytes =
                                static_cast<size_t>(ROCM_NATIVE_SMALL_M_GRAPH_SAFE_KB_CAP) *
                                static_cast<size_t>(n) * sizeof(float);
                            const size_t stream_slot_bytes =
                                static_cast<size_t>(ROCM_NATIVE_SMALL_M_GRAPH_SAFE_KB_CAP) *
                                scatter_slot_columns[static_cast<size_t>(stream_idx)] *
                                sizeof(float);
                            if (required_partial_bytes > stream_slot_bytes)
                            {
                                throw std::runtime_error(
                                    "[ConcurrentDecode] Batched scatter workspace "
                                    "slot is too small for projection " +
                                    std::to_string(pi));
                            }
                            auto *stream_partial =
                                reinterpret_cast<float *>(
                                    reinterpret_cast<unsigned char *>(
                                        impl_->d_scatter_partial_batched) +
                                    scatter_slot_offsets[
                                        static_cast<size_t>(stream_idx)]);

                            proj_ok = rocmGemv_native_quantised_fp32_with_policy(
                                impl_->d_A_int8,
                                rocm_kernel->impl_->d_weights_native_quantised,
                                rocm_kernel->impl_->d_weights_native_scales,
                                rocm_kernel->impl_->d_weights_native_mins,
                                rocm_kernel->impl_->d_weights_native_emins,
                                d_output,
                                impl_->d_scales_A,
                                stream_partial,
                                n, k,
                                rocm_kernel->impl_->native_quantised_codebook_id,
                                requireNativeQuantisedArithmeticPolicyCodebook(
                                    rocm_kernel->impl_->native_quantised_codebook_id,
                                    rocm_kernel->impl_->native_source_identity),
                                rocm_device_id_, pool.streams[stream_idx],
                                fused_uses_blockwise_shared_quant ? impl_->d_scales_A_blockwise : nullptr);
                        }
                        else
                        {
                            // INT8-VNNI GEMV: no shared scratch needed
                            int8_t *d_vnni = rocm_kernel->impl_->d_weights_int8_vnni;
                            if (!d_vnni)
                            {
                                throw std::runtime_error(
                                    "[ConcurrentDecode] Projection " + std::to_string(pi) +
                                    " has no INT8-VNNI weights — kernel not prepared");
                            }

                            proj_ok = rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                                impl_->d_A_int8, d_vnni, d_output,
                                impl_->d_scales_A_blockwise, d_scales_B,
                                n, k,
                                1.0f, 0.0f,
                                nullptr, d_bias,
                                rocm_device_id_, pool.streams[stream_idx]);
                        }

                        if (!proj_ok)
                        {
                            throw std::runtime_error(
                                "[ConcurrentDecode] Projection " + std::to_string(pi) +
                                " GEMV kernel launch failed on stream " + std::to_string(pi % pool.count) +
                                " — cannot continue inference");
                        }

                        if (rocm_kernel->impl_->has_native_quantised && d_bias)
                        {
                            if (!rocmQuantGemm_biasAdd(d_output, d_bias, m, n,
                                                       rocm_device_id_, pool.streams[stream_idx]))
                            {
                                throw std::runtime_error(
                                    "[ConcurrentDecode] Bias add failed for projection " +
                                    std::to_string(pi) + " on stream " + std::to_string(stream_idx));
                            }
                        }

                        (void)hipEventRecord(pool.completion[stream_idx],
                                       pool.streams[stream_idx]);
                    }

                    // All projections dispatched — main stream waits for completion
                    for (int si = 0; si < std::min(num_proj, pool.count); ++si)
                    {
                        (void)hipStreamWaitEvent(
                            static_cast<hipStream_t>(gpu_stream_),
                            pool.completion[si], 0);
                    }

                    /* Prove the persistent side-stream decode branch ran. */
            if (PerfStatsCollector::isDomainEnabled("kernel"))
                    {
                        PerfStatsCollector::addCounter(
                            "kernel",
                            "rocm_fused_projection_stream_pool_calls",
                            1.0,
                            "gemm",
                            "rocm:" + std::to_string(rocm_device_id_),
                            PerfStatsCollector::Tags{
                                {"mode", "decode"},
                                {"m", std::to_string(m)},
                                {"k", std::to_string(k)},
                                {"projections", std::to_string(num_proj)},
                                {"streams", std::to_string(
                                     std::min(num_proj, pool.count))}});
                    }

                    LOG_TRACE("[ConcurrentDecode] All " << num_proj
                                                        << " projections dispatched concurrently");

                    if (ws && ws != saved_workspace)
                        workspace_ = saved_workspace;
                    publish_projection_outputs();
                    return true;
                }
            }
#endif // HAVE_ROCM

#ifdef HAVE_ROCM
            if (small_m_native_verifier_fused &&
                projections.size() >= 2 &&
                projections.size() <= 8)
            {
                std::array<const uint8_t *, 8> payloads{};
                std::array<const uint16_t *, 8> scales{};
                std::array<const uint16_t *, 8> mins{};
                std::array<const uint32_t *, 8> emins{};
                std::array<float *, 8> outputs{};
                std::array<float *, 8> partials{};
                std::array<const float *, 8> biases{};
                std::array<int, 8> Ns{};
                std::array<uint8_t, 8> codebooks{};

                bool batched_eligible = true;
                std::string batched_bypass_reason;
                uint8_t common_codebook = 0;
                bool have_common_codebook = false;
                bool mixed_codebooks = false;

                auto mark_batched_bypass = [&](std::string reason)
                {
                    if (batched_bypass_reason.empty())
                        batched_bypass_reason = std::move(reason);
                    batched_eligible = false;
                };

                for (size_t i = 0; i < projections.size(); ++i)
                {
                    const auto &proj = projections[i];
                    auto original_scope = proj.beginOriginalOutputSliceScope();
                    auto *rocm_kernel = dynamic_cast<ROCmQuantisedGemmKernel *>(proj.kernel);
                    auto *fp32_output = dynamic_cast<FP32Tensor *>(proj.output);
                    if (!rocm_kernel || !rocm_kernel->impl_ || !rocm_kernel->impl_->has_native_quantised)
                    {
                        mark_batched_bypass("non_native_projection");
                        break;
                    }
                    if (!fp32_output)
                    {
                        mark_batched_bypass("non_fp32_output");
                        break;
                    }
                    if (fp32_output->isMapped())
                    {
                        mark_batched_bypass("mapped_output");
                        break;
                    }

                    try
                    {
                        rocm_kernel->validateWorkspace();
                    }
                    catch (const std::exception &ex)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                  << " cannot validate graph workspace for batched small-M route: "
                                  << ex.what());
                        return false;
                    }

                    if (!rocm_kernel->impl_->d_weights_native_quantised ||
                        !rocm_kernel->impl_->d_weights_native_scales ||
                        !rocm_kernel->impl_->d_scatter_partial)
                    {
                        mark_batched_bypass("missing_native_buffers");
                        break;
                    }
                    if (fp32_output->rows() < static_cast<size_t>(m) ||
                        fp32_output->cols() < static_cast<size_t>(proj.n))
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                  << " output capacity too small for graph-native batched small-M route"
                                  << " rows=" << fp32_output->rows()
                                  << " cols=" << fp32_output->cols()
                                  << " required_rows=" << m
                                  << " required_cols=" << proj.n);
                        return false;
                    }

                    const uint8_t codebook = rocm_kernel->impl_->native_quantised_codebook_id;
                    if (!have_common_codebook)
                    {
                        common_codebook = codebook;
                        have_common_codebook = true;
                    }
                    else if (common_codebook != codebook)
                    {
                        mixed_codebooks = true;
                    }

                    float *d_output = static_cast<float *>(fp32_output->gpu_data_ptr());
                    if (!d_output)
                    {
                        mark_batched_bypass("missing_output_device_ptr");
                        break;
                    }

                    const float *d_bias = nullptr;
                    if (proj.bias)
                    {
                        if (proj.bias->native_type() != TensorType::FP32)
                        {
                            mark_batched_bypass("non_fp32_bias");
                            break;
                        }

                        auto *bias_tensor = const_cast<TensorBase *>(proj.bias);
                        const DeviceId target_device = DeviceId::rocm(rocm_device_id_);
                        const auto current_dev = bias_tensor->current_device();
                        if (current_dev.has_value() &&
                            current_dev->is_gpu() &&
                            current_dev.value() != target_device)
                        {
                            mark_batched_bypass("bias_wrong_gpu");
                            break;
                        }
                        TransferEngine::requireDeviceInput(
                            bias_tensor, target_device, gpu_stream_);
                        d_bias = static_cast<const float *>(bias_tensor->gpu_data_ptr());

                        if (!d_bias)
                        {
                            mark_batched_bypass("missing_bias_device_ptr");
                            break;
                        }
                    }

                    payloads[i] = rocm_kernel->impl_->d_weights_native_quantised;
                    scales[i] = static_cast<const uint16_t *>(rocm_kernel->impl_->d_weights_native_scales);
                    mins[i] = static_cast<const uint16_t *>(rocm_kernel->impl_->d_weights_native_mins);
                    emins[i] = static_cast<const uint32_t *>(rocm_kernel->impl_->d_weights_native_emins);
                    outputs[i] = d_output;
                    partials[i] = nullptr;
                    biases[i] = d_bias;
                    Ns[i] = proj.n;
                    codebooks[i] = codebook;

                    LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Small-M batched projection "
                              << i
                              << " name=" << (proj.name ? proj.name : "?")
                              << " kernel=" << static_cast<const void *>(rocm_kernel)
                              << " output=" << static_cast<const void *>(outputs[i])
                              << " n=" << proj.n);
                }

                if (batched_eligible)
                {
                    const int expected_kb = computeNativeQuantisedSmallMBatchedKBForValidation(
                        Ns,
                        static_cast<int>(projections.size()),
                        m,
                        k);
                    if (static_cast<int>(projections.size()) > ROCM_NATIVE_SMALL_M_WORKSPACE_BATCH_PROJECTIONS)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] NativeQuantised fused small-M batched route "
                                  << "requires projection count <= "
                                  << ROCM_NATIVE_SMALL_M_WORKSPACE_BATCH_PROJECTIONS
                                  << " for declared graph workspace, got " << projections.size());
                        return false;
                    }

                    float *batched_partial_base = impl_->d_scatter_partial_batched;
                    const size_t batched_partial_bytes =
                        workspace_
                            ? workspace_->getBufferSize(GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL_BATCHED)
                            : 0;
                    size_t partial_offset_floats = 0;

                    for (size_t i = 0; i < projections.size(); ++i)
                    {
                        const size_t required_projection_floats =
                            static_cast<size_t>(std::max(1, expected_kb)) *
                            // The launch reuses one physical tile for wider M;
                            // match the declared arena instead of pricing all
                            // logical rows as simultaneously resident partials.
                            static_cast<size_t>(std::min(
                                m, kDefaultNativeQuantisedVerifierRowCapacity)) *
                            static_cast<size_t>(Ns[i]);
                        const size_t next_offset = partial_offset_floats + required_projection_floats;
                        if (!batched_partial_base ||
                            next_offset * sizeof(float) > batched_partial_bytes)
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Batched split-K partial arena "
                                      << "is missing or too small"
                                      << " bytes=" << batched_partial_bytes
                                      << " required=" << (next_offset * sizeof(float))
                                      << " expected_kb=" << expected_kb
                                      << " m=" << m
                                      << " projection=" << i
                                      << " n=" << Ns[i]
                                      << " projections=" << projections.size());
                            return false;
                        }
                        partials[i] = batched_partial_base + partial_offset_floats;
                        partial_offset_floats = next_offset;

                        for (size_t j = i + 1; j < projections.size(); ++j)
                        {
                            if (outputs[i] == outputs[j])
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projections "
                                          << i << " and " << j
                                          << " share one output buffer in graph-native batched small-M route"
                                          << " output=" << static_cast<const void *>(outputs[i]));
                                return false;
                            }
                            if (partials[i] == partials[j])
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projections "
                                          << i << " and " << j
                                          << " share one split-K partial buffer in graph-native batched small-M route"
                                          << " partial=" << static_cast<const void *>(partials[i])
                                          << " expected_kb=" << expected_kb
                                          << ". Each projection must receive a distinct slice from "
                                          << GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL_BATCHED
                                          << ".");
                                return false;
                            }
                        }

                        LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Small-M batched projection "
                                  << i
                                  << " partial=" << static_cast<const void *>(partials[i])
                                  << " arena=" << GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL_BATCHED
                                  << " offset_floats=" << (partial_offset_floats - required_projection_floats)
                                  << " n=" << Ns[i]);
                    }

                    auto launch_group = [&](const std::vector<int> &indices,
                                            int policy_kb = -1,
                                            int policy_target_waves = -1) -> bool
                    {
                        std::array<const uint8_t *, 8> group_payloads{};
                        std::array<const uint16_t *, 8> group_scales{};
                        std::array<const uint16_t *, 8> group_mins{};
                        std::array<const uint32_t *, 8> group_emins{};
                        std::array<const float *, 8> group_biases{};
                        std::array<float *, 8> group_outputs{};
                        std::array<float *, 8> group_partials{};
                        std::array<int, 8> group_Ns{};
                        std::array<uint8_t, 8> group_codebooks{};

                        bool group_mixed = false;
                        const uint8_t group_codebook = codebooks[indices.front()];
                        for (size_t group_slot = 0; group_slot < indices.size(); ++group_slot)
                        {
                            const int projection_index = indices[group_slot];
                            group_payloads[group_slot] = payloads[projection_index];
                            group_scales[group_slot] = scales[projection_index];
                            group_mins[group_slot] = mins[projection_index];
                            group_emins[group_slot] = emins[projection_index];
                            group_biases[group_slot] = biases[projection_index];
                            group_outputs[group_slot] = outputs[projection_index];
                            group_partials[group_slot] = partials[projection_index];
                            group_Ns[group_slot] = Ns[projection_index];
                            group_codebooks[group_slot] = codebooks[projection_index];
                            group_mixed = group_mixed || (codebooks[projection_index] != group_codebook);
                        }

                        const int group_count = static_cast<int>(indices.size());
                        return group_mixed
                            ? rocmGemv_native_quantised_small_m_batched_mixed_fp32_with_sums_policy(
                                  impl_->d_A_int8,
                                  group_payloads.data(),
                                  group_scales.data(),
                                  group_mins.data(),
                                  group_emins.data(),
                                  group_biases.data(),
                                  group_outputs.data(),
                                  impl_->d_scales_A_blockwise,
                                  impl_->d_sums_A_blockwise,
                                  group_partials.data(),
                                  group_Ns.data(),
                                  group_codebooks.data(),
                                  group_count,
                                  m, k,
                                  rocm_device_id_,
                                  gpu_stream_,
                                  policy_kb,
                                  policy_target_waves,
                            VerifierKernelModeScope::rowsFor(m))
                            : rocmGemv_native_quantised_small_m_batched_fp32_with_sums_policy(
                                  impl_->d_A_int8,
                                  group_payloads.data(),
                                  group_scales.data(),
                                  group_mins.data(),
                                  group_emins.data(),
                                  group_biases.data(),
                                  group_outputs.data(),
                                  impl_->d_scales_A_blockwise,
                                  impl_->d_sums_A_blockwise,
                                  group_partials.data(),
                                  group_Ns.data(),
                                  group_count,
                                  m, k,
                                  group_codebook,
                                  rocm_device_id_,
                                  gpu_stream_,
                                  policy_kb,
                                  policy_target_waves,
                            VerifierKernelModeScope::rowsFor(m));
                    };

                    bool gemv_ok = true;
                    if (g_rocm_native_quantised_decode_equivalent_scope)
                    {
                        struct GeneratedGroupPolicy
                        {
                            int kb = -1;
                            int target_waves = -1;
                        };

                        std::vector<std::vector<int>> groups;
                        std::vector<GeneratedGroupPolicy> group_policies;
                        bool generated_grouping_available = true;
                        for (size_t i = 0; i < projections.size(); ++i)
                        {
                            auto original_scope = projections[i].beginOriginalOutputSliceScope();
                            int policy_kb = -1;
                            int policy_target_waves = -1;
                            /*
                             * Decode-equivalent verifier publication is a
                             * state-commit path, not a generic M-aware GEMV
                             * throughput path. Grouping projections by the
                             * grouped-M table can choose a different split-K
                             * policy than the row-by-row decode oracle, which
                             * changes FP32 reduction order and breaks strict
                             * row equivalence.
                             *
                             * Resolve and pass the serial-M1 policy explicitly
                             * even when this call contains only one projection.
                             * ROCm LocalTP invokes this method concurrently from
                             * one host worker per device. The HIP launcher's
                             * compatibility mode bit is process-global, so it
                             * cannot be the authority for a per-worker verifier
                             * transaction: one worker may leave its scope while
                             * another is still dispatching. Explicit launch
                             * parameters make the grouped operation independent
                             * of that shared compatibility bit and preserve the
                             * serial reduction geometry by construction.
                             */
                            const bool policy_ready = projections[i].original_output_slice
                                ? rocmGemv_native_quantised_query_serial_m1_config_with_policy(
                                    canonicalDeviceVnniCodebookId(projections[i].original_output_slice->sourceIdentity().codebook_id),
                                    Ns[i], k, &policy_kb, &policy_target_waves)
                                : rocmGemv_native_quantised_query_serial_m1_config(
                                    codebooks[i], Ns[i], k, &policy_kb, &policy_target_waves);
                            if (!policy_ready)
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection "
                                          << i
                                          << " has no serial-M1 NativeQuantised policy for decode-equivalent grouped verifier"
                                          << " codebook=" << static_cast<int>(codebooks[i])
                                          << " m=" << m
                                          << " n=" << Ns[i]
                                          << " k=" << k
                                          << "; refusing serial-M1 fallback. Refresh the ROCm NativeQuantised decode dispatch table.");
                                PerfStatsCollector::addCounter(
                                    "kernel",
                                    "rocm_native_quantised_small_m_generated_policy_miss",
                                    1.0,
                                    "gemm",
                                    "rocm:" + std::to_string(rocm_device_id_),
                                    PerfStatsCollector::Tags{
                                        {"m", std::to_string(m)},
                                        {"n", std::to_string(Ns[i])},
                                        {"k", std::to_string(k)},
                                        {"codebook", std::to_string(static_cast<int>(codebooks[i]))}});
                                generated_grouping_available = false;
                                break;
                            }

                            int group_index = -1;
                            for (size_t group = 0; group < group_policies.size(); ++group)
                            {
                                if (group_policies[group].kb == policy_kb &&
                                    group_policies[group].target_waves == policy_target_waves)
                                {
                                    group_index = static_cast<int>(group);
                                    break;
                                }
                            }
                            if (group_index < 0)
                            {
                                group_policies.push_back(GeneratedGroupPolicy{
                                    policy_kb,
                                    policy_target_waves});
                                groups.emplace_back();
                                group_index = static_cast<int>(groups.size() - 1);
                            }
                            groups[static_cast<size_t>(group_index)].push_back(static_cast<int>(i));
                        }

                        if (!generated_grouping_available)
                        {
                            return false;
                        }

                        for (size_t group_index = 0; group_index < groups.size(); ++group_index)
                        {
                            const GeneratedGroupPolicy &policy = group_policies[group_index];
                            gemv_ok = gemv_ok && launch_group(
                                                     groups[group_index],
                                                     policy.kb,
                                                     policy.target_waves);
                        }
                    }
                    else
                    {
                        std::vector<int> all_indices;
                        all_indices.reserve(projections.size());
                        for (size_t i = 0; i < projections.size(); ++i)
                            all_indices.push_back(static_cast<int>(i));
                        gemv_ok = launch_group(all_indices);
                    }
                    if (!gemv_ok)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] NativeQuantised fused small-M batched GEMV failed"
                                  << " m=" << m << " k=" << k
                                  << " projections=" << projections.size()
                                  << " codebook=" << (mixed_codebooks ? std::string("mixed") : std::to_string(static_cast<int>(common_codebook))));
                        return false;
                    }

        if (PerfStatsCollector::isDomainEnabled("kernel"))
                    {
                        PerfStatsCollector::addCounter(
                            "kernel",
                            "rocm_native_quantised_small_m_batched_calls",
                            1.0,
                            "gemm",
                            "rocm:" + std::to_string(rocm_device_id_),
                            PerfStatsCollector::Tags{
                                {"m", std::to_string(m)},
                                {"codebook", mixed_codebooks ? std::string("mixed") : std::to_string(static_cast<int>(common_codebook))},
                                {"k", std::to_string(k)},
                                {"projections", std::to_string(projections.size())}});

                        if (mixed_codebooks)
                        {
                            PerfStatsCollector::addCounter(
                                "kernel",
                                "rocm_native_quantised_small_m_batched_mixed_calls",
                                1.0,
                                "gemm",
                                "rocm:" + std::to_string(rocm_device_id_),
                                PerfStatsCollector::Tags{
                                    {"m", std::to_string(m)},
                                    {"k", std::to_string(k)},
                                    {"projections", std::to_string(projections.size())}});
                        }

                        for (size_t projection_index = 0; projection_index < projections.size(); ++projection_index)
                        {
                            const int n_value = Ns[projection_index];
                            if (n_value <= 0)
                                continue;
                            PerfStatsCollector::addCounter(
                                "kernel",
                                "rocm_native_quantised_small_m_batched_projection_calls",
                                1.0,
                                "gemm",
                                "rocm:" + std::to_string(rocm_device_id_),
                                PerfStatsCollector::Tags{
                                    {"m", std::to_string(m)},
                                    {"codebook", std::to_string(static_cast<int>(codebooks[projection_index]))},
                                    {"n", std::to_string(n_value)},
                                    {"k", std::to_string(k)}});
                        }
                    }

                    LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Batched native small-M verifier complete"
                              << " M=" << m << " K=" << k
                              << " projections=" << projections.size());
                    publish_projection_outputs();
                    return true;
                }

            if (PerfStatsCollector::isDomainEnabled("kernel") &&
                !batched_bypass_reason.empty())
                {
                    PerfStatsCollector::addCounter(
                        "kernel",
                        "rocm_native_quantised_small_m_batched_bypasses",
                        1.0,
                        "gemm",
                        "rocm:" + std::to_string(rocm_device_id_),
                        PerfStatsCollector::Tags{
                            {"m", std::to_string(m)},
                            {"k", std::to_string(k)},
                            {"projections", std::to_string(projections.size())},
                            {"reason", batched_bypass_reason}});
                }
            }
#endif // HAVE_ROCM

            // Ordinary direct-device M1 owns a static grouped-reduction implementation.
            // Mapped destinations retain their existing serialized copy-out transaction:
            // its one shared d_C_fp32 cannot back simultaneous pending producers.
            const bool direct_native_m1 = m==1 && !countedVerifierRows(m) && projections.size()>=2 && all_projections_native_quantised &&
                std::all_of(projections.begin(),projections.end(),[](const auto &p)
                {
                    const auto *output=dynamic_cast<const FP32Tensor *>(p.output);
                    return output && !output->isMapped();
                });
            if(direct_native_m1)
            {
                constexpr std::size_t slots=quantized_gemm_workspace::kFusedProjectionStreamCapacity;
                constexpr std::size_t capacity=quantized_gemm_workspace::kScatterKBlockCapacity;
                std::array<std::size_t,slots> widths{},offsets{};
                const auto slot_count=std::min(projections.size(),slots);
                // Original modulo-slot assignment is the workspace ABI, independent
                // of format/geometry buckets selected later by reduction submission.
                for(std::size_t i=0;i<projections.size();++i)
                {
                    if(projections[i].n<=0)throw std::invalid_argument("Grouped M1 projection width must be positive");
                    widths[i%slot_count]=std::max(widths[i%slot_count],std::size_t(projections[i].n));
                }
                std::size_t required=0;
                for(std::size_t slot=0;slot<slot_count;++slot)
                {
                    offsets[slot]=required;
                    const auto bytes=quantized_gemm_workspace::checkedProduct(
                        quantized_gemm_workspace::checkedProduct(capacity,widths[slot],"M1 slot columns"),sizeof(float),"M1 slot bytes");
                    if(bytes>std::numeric_limits<std::size_t>::max()-required)
                        throw std::invalid_argument("Grouped M1 partial slot prefix overflows");
                    required+=bytes;
                }
                const auto workspace_name=scatterPartialBatchedBufferName();
                if(!impl_->d_scatter_partial_batched || !workspace_ || workspace_->getBufferSize(workspace_name)<required)
                    throw std::invalid_argument("Grouped M1 requires its complete canonical partial slot arena");
                join_projection_biases_to_root("GroupedM1Reduction");
                const auto interval=[](const void *pointer,std::size_t bytes)
                {
                    const auto address=reinterpret_cast<std::uintptr_t>(pointer);
                    if(!address || !bytes || bytes>std::numeric_limits<std::uintptr_t>::max()-address)
                        throw std::invalid_argument("Grouped M1 live operand interval is absent or overflows");
                    return std::pair{address,address+bytes};
                };
                const auto overlaps=[](auto a,auto b){return a.first<b.second && b.first<a.second;};
                const auto partial_bank=interval(impl_->d_scatter_partial_batched,required);
                // Prove the entire output/bias closure before any producer runs.
                for(std::size_t i=0;i<projections.size();++i)
                {
                    const auto &p=projections[i];
                    auto *kernel=dynamic_cast<ROCmQuantisedGemmKernel *>(p.kernel);
                    if(!kernel || !kernel->impl_ || !kernel->impl_->has_native_quantised ||
                        kernel->rocm_device_id_!=rocm_device_id_ || kernel->N_!=p.n || kernel->K_!=k ||
                        p.output->size_bytes()<std::size_t(p.n)*sizeof(float))
                        throw std::invalid_argument("Grouped M1 requires complete original prepared/output owners");
                    const auto output_device=p.output->current_device();
                    if(output_device && output_device->is_gpu() && *output_device!=target_device)
                        throw std::invalid_argument("Grouped M1 output belongs to another device");
                    TransferEngine::requireDeviceOutput(p.output,target_device,gpu_stream_);
                    const auto output=interval(p.output->gpu_data_ptr(),std::size_t(p.n)*sizeof(float));
                    if(overlaps(output,partial_bank))throw std::invalid_argument("Grouped M1 output aliases its partial arena");
                    for(std::size_t j=0;j<projections.size();++j)
                    {
                        const auto &other=projections[j];
                        if(i!=j && overlaps(output,interval(other.output->gpu_data_ptr(),std::size_t(other.n)*sizeof(float))))
                            throw std::invalid_argument("Grouped M1 output owners alias");
                        if(other.bias)
                        {
                            if(other.bias->size_bytes()<std::size_t(other.n)*sizeof(float))
                                throw std::invalid_argument("Grouped M1 bias owner is truncated");
                            const auto bias=interval(other.bias->gpu_data_ptr(),std::size_t(other.n)*sizeof(float));
                            if(overlaps(output,bias) || overlaps(partial_bank,bias))
                                throw std::invalid_argument("Grouped M1 output/partial writers alias immutable bias");
                        }
                    }
                    kernel->validateWorkspace();
                }
                for(std::size_t first=0;first<projections.size();first+=slot_count)
                {
                    // Tickets cannot move/copy, and die only after explicit submission.
                    // Retire every wave's reducers before reusing any original slot.
                    std::array<NativeM1ReductionTicket,slots> tickets;
                    const auto count=std::min(slot_count,projections.size()-first);
                    for(std::size_t local=0;local<count;++local)
                    {
                        const auto &p=projections[first+local];
                        auto original_scope=p.beginOriginalOutputSliceScope();
                        auto *kernel=static_cast<ROCmQuantisedGemmKernel *>(p.kernel);
                        auto *partial=reinterpret_cast<float *>(reinterpret_cast<unsigned char *>(impl_->d_scatter_partial_batched)+offsets[local]);
                        if(!enqueueNativeM1Producer(tickets[local],impl_->d_A_int8,
                            kernel->impl_->d_weights_native_quantised,kernel->impl_->d_weights_native_scales,
                            kernel->impl_->d_weights_native_mins,kernel->impl_->d_weights_native_emins,
                            static_cast<float *>(p.output->gpu_data_ptr()),impl_->d_scales_A,partial,p.n,k,
                            kernel->impl_->native_quantised_codebook_id,
                            requireNativeQuantisedArithmeticPolicyCodebook(kernel->impl_->native_quantised_codebook_id,kernel->impl_->native_source_identity),
                            rocm_device_id_,gpu_stream_,fused_uses_blockwise_shared_quant?impl_->d_scales_A_blockwise:nullptr))
                            throw std::runtime_error("Grouped M1 original producer submission failed");
                    }
                    if(!submitNativeM1Reductions(std::span(tickets.data(),count),rocm_device_id_,gpu_stream_))
                        throw std::runtime_error("Grouped M1 reduction submission failed");
                    // Preserve the original standalone bias operation AFTER publication;
                    // no fused +0 or changed association enters the ordinary RN fold.
                    for(std::size_t local=0;local<count;++local)
                    {
                        const auto &p=projections[first+local];
                        if(p.bias && !rocmQuantGemm_biasAdd(static_cast<float *>(p.output->gpu_data_ptr()),
                            static_cast<const float *>(p.bias->gpu_data_ptr()),1,p.n,rocm_device_id_,gpu_stream_))return false;
                    }
                }
                if(PerfStatsCollector::isDomainEnabled("kernel"))
                    PerfStatsCollector::addCounter("kernel","rocm_native_m1_grouped_reduction_calls",1.0,"gemm",
                        "rocm:"+std::to_string(rocm_device_id_),PerfStatsCollector::Tags{{"projections",std::to_string(projections.size())}});
                publish_projection_outputs();return true;
            }

            for (size_t i = 0; i < projections.size() && all_success; ++i)
            {
                const auto &proj = projections[i];
                auto original_scope = proj.beginOriginalOutputSliceScope();
                if (!proj.kernel || !proj.output)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i << " has null kernel or output");
                    all_success = false;
                    break;
                }

                // Get the ROCm kernel for this projection
                auto *rocm_kernel = dynamic_cast<ROCmQuantisedGemmKernel *>(proj.kernel);
                if (!rocm_kernel)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                             << " kernel is not a ROCmQuantisedGemmKernel");
                    all_success = false;
                    break;
                }

                const int n = proj.n;
                LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                         << " (" << (proj.name ? proj.name : "unnamed") << "): m=" << m << " n=" << n << " k=" << k);

                // Weights already converted in Step 3 (ensureWeightsConverted loop).
                // Validate this projection's workspace is bound and populated.
                rocm_kernel->validateWorkspace();

                // Ensure output tensor is on device
                if (proj.output->native_type() != TensorType::FP32)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                             << " output must be FP32, got " << static_cast<int>(proj.output->native_type()));
                    all_success = false;
                    break;
                }

                auto *fp32_output = dynamic_cast<FP32Tensor *>(proj.output);
                if (!fp32_output)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Failed to cast output to FP32Tensor");
                    all_success = false;
                    break;
                }

                // Coherence handled automatically by DeviceGraphExecutor
                float *d_output = static_cast<float *>(fp32_output->gpu_data_ptr());
                if (!d_output)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Output has no GPU data for projection " << i);
                    all_success = false;
                    break;
                }

                // Get weight scales from this projection's kernel
                float *d_scales_B = nullptr;

                if (rocm_kernel->packed_)
                {
                    d_scales_B = rocm_kernel->packed_->d_scales;
                }
                else if (rocm_kernel->impl_)
                {
                    d_scales_B = rocm_kernel->impl_->d_scales_B;
                }

                // NativeQuantised path uses impl_->d_weights_native_scales (not d_scales_B),
                // so d_scales_B may be null when only native-quantised weights are present.
                const bool native_quantised_fused = rocm_kernel->impl_ && rocm_kernel->impl_->has_native_quantised;
                if (!d_scales_B && !native_quantised_fused)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i << " weight scales not on device");
                    all_success = false;
                    break;
                }

                // Resolve bias pointer early (needed by both GEMV and CK paths)
                const float *d_bias = nullptr;
                if (proj.bias)
                {
                    if (proj.bias->native_type() != TensorType::FP32)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                                 << " bias must be FP32, got " << static_cast<int>(proj.bias->native_type()));
                        all_success = false;
                        break;
                    }

                    auto *bias_tensor = const_cast<TensorBase *>(proj.bias);
                    DeviceId target_device = DeviceId::rocm(rocm_device_id_);
                    auto current_dev = bias_tensor->current_device();

                    LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Proj " << i
                                                                                       << " bias tensor=" << bias_tensor
                                                                                       << " current_dev=" << (current_dev.has_value() ? current_dev->to_string() : "(none)")
                                                                                       << " target_device=" << target_device.to_string()
                                                                                       << " gpu_data_ptr=" << bias_tensor->gpu_data_ptr()
                                                                                       << " rocm_device_id_=" << rocm_device_id_);

                    if (current_dev.has_value() &&
                        current_dev->is_gpu() &&
                        current_dev.value() != target_device)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] MULTI-GPU CONFLICT: Bias tensor is on "
                                  << current_dev->to_string() << " but we need ROCm:" << rocm_device_id_
                                  << ". Ensure WeightPreloader::uploadNonGemmWeights() was called for this device.");
                        all_success = false;
                        break;
                    }
                    TransferEngine::requireDeviceInput(
                        bias_tensor, target_device, gpu_stream_);
                    d_bias = static_cast<const float *>(bias_tensor->gpu_data_ptr());

                    if (!d_bias)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Bias has no GPU data for projection " << i);
                        all_success = false;
                        break;
                    }
                    LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                             << " using bias ptr=" << static_cast<const void *>(d_bias));
                }

                // =========================================================================
                // DECODE FAST PATH: M=1 GEMV (skips CK GEMM entirely)
                //
                // For M=1 (single-token decode), we use a FUSED kernel that combines
                // FP32→INT8 quantization + INT8×INT8 GEMV + FP32 scaling into a single
                // kernel launch. This eliminates the entire 3-kernel pipeline:
                //   1. quantizeActivationsQ8_kernel (eliminated - done in shared memory)
                //   2. GEMV INT8×INT8→INT32 (fused)
                //   3. applyScaling INT32→FP32 (eliminated - scale applied in-kernel)
                //
                // The fused kernel:
                //   - Reads FP32 activations directly (no pre-quantization needed)
                //   - Quantizes per-tile (256 elements) in shared memory via warp reduction
                //   - Computes INT8 dot products using v_dot4_i32_i8
                //   - Applies scale_A * scale_B and atomicAdd's FP32 result to output
                //   - Bias applied separately (atomicAdd race prevents fusion)
                //
                // Result: 1 kernel instead of 3, eliminating ~3.6ms/token pipeline overhead.
                // =========================================================================
                if (m == 1 && !countedVerifierRows(m))
                {
                    int8_t *d_vnni = rocm_kernel->impl_ ? rocm_kernel->impl_->d_weights_int8_vnni : nullptr;
                    if (!rocm_kernel->impl_->has_native_quantised && !d_vnni)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i << " VNNI weights not on device");
                        all_success = false;
                        break;
                    }

                    // NativeQuantised path: lossless Q4/IQ4 decode with typed block metadata
                    if (rocm_kernel->impl_->has_native_quantised)
                    {
                        const bool output_is_mapped = fp32_output->isMapped();
                        const bool output_needs_copyout = output_is_mapped;
                        float *d_native_output = output_needs_copyout ? impl_->d_C_fp32 : d_output;

                        // Activations are always pre-quantized above (Step 3)
                        LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                                 << " NATIVE-QUANTISED GEMV M=1 N=" << n << " K=" << k
                                                                                                 << (d_bias ? " +bias" : ""));

                        if (!rocmGemv_native_quantised_fp32_with_policy(
                                impl_->d_A_int8,
                                rocm_kernel->impl_->d_weights_native_quantised,
                                rocm_kernel->impl_->d_weights_native_scales,
                                rocm_kernel->impl_->d_weights_native_mins,
                                rocm_kernel->impl_->d_weights_native_emins,
                                d_native_output,
                                impl_->d_scales_A,
                                impl_->d_scatter_partial,
                                n, k,
                                rocm_kernel->impl_->native_quantised_codebook_id,
                                requireNativeQuantisedArithmeticPolicyCodebook(
                                    rocm_kernel->impl_->native_quantised_codebook_id,
                                    rocm_kernel->impl_->native_source_identity),
                                rocm_device_id_, gpu_stream_,
                                fused_uses_blockwise_shared_quant ? impl_->d_scales_A_blockwise : nullptr))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] NativeQuantised GEMV failed for projection " << i);
                            all_success = false;
                            break;
                        }

                        if (d_bias)
                        {
                            if (!rocmQuantGemm_biasAdd(d_native_output, d_bias, m, n, rocm_device_id_, gpu_stream_))
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] NativeQuantised bias add failed for projection " << i);
                                all_success = false;
                                break;
                            }
                        }

                        // Copy back from workspace to d_output
                        if (output_needs_copyout)
                        {
                            const size_t output_bytes = static_cast<size_t>(m) * n * sizeof(float);
                            const hipError_t copy_err = hipMemcpyAsync(
                                d_output,
                                d_native_output,
                                output_bytes,
                                hipMemcpyDeviceToDevice,
                                static_cast<hipStream_t>(gpu_stream_));
                            if (copy_err != hipSuccess)
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Device-alias GEMV publication failed: "
                                          << hipGetErrorString(copy_err));
                                all_success = false;
                                break;
                            }
                        }

                        LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i << " native-quantised GEMV complete");
                        continue;
                    }

                    // BATCHED INT8 SCATTER PATH: collect pointers for single batched dispatch
                    // All VNNI projections with the same K are batched into 2 kernel launches
                    // (1 batched scatter + 1 batched reduce) instead of 2N individual launches.
                    if (d_vnni && batch_count < 8)
                    {
                        LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                                 << " (" << (proj.name ? proj.name : "unnamed")
                                                                                                 << ") BATCHED SCATTER collect: N=" << n << " K=" << k
                                                                                                 << (d_bias ? " +bias" : ""));
                        batch_B_ptrs[batch_count] = d_vnni;
                        batch_C_ptrs[batch_count] = d_output;
                        batch_scales_B_ptrs[batch_count] = d_scales_B;
                        batch_bias_ptrs[batch_count] = d_bias;
                        batch_N[batch_count] = n;
                        batch_count++;
                        continue; // Dispatch happens after the loop
                    }

                    // Fallback: single-projection INT8 scatter (batch overflow or no VNNI)
                    if (d_vnni)
                    {
                        LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                                 << " SINGLE INT8 SCATTER M=1 N=" << n << " K=" << k);
                        bool projection_ok = rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                            impl_->d_A_int8, d_vnni, d_output,
                            impl_->d_scales_A_blockwise, d_scales_B,
                            n, k,
                            1.0f, 0.0f,
                            nullptr, d_bias,
                            rocm_device_id_, gpu_stream_);

                        if (!projection_ok)
                        {
                            projection_ok = rocmGemv_int8_scatter_vnni_blockwise(
                                impl_->d_A_int8, d_vnni, d_output,
                                impl_->d_scales_A_blockwise, d_scales_B, d_bias,
                                rocm_kernel->impl_->d_scatter_partial, rocm_kernel->impl_->d_selfreduce_counters,
                                n, k, 1.0f, 0.0f, nullptr,
                                rocm_device_id_, gpu_stream_);
                        }

                        if (!projection_ok)
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] INT8 scatter GEMV failed for projection " << i);
                            all_success = false;
                            break;
                        }
                    }
                    else
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                                 << " has no VNNI weights and is not native-quantised");
                        all_success = false;
                        break;
                    }

                    LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i << " GEMV complete");
                    continue; // Skip CK path below
                }

                // =========================================================================
                // PREFILL PATH: M>1 CK GEMM
                // =========================================================================

                const bool output_is_mapped = fp32_output->isMapped();
                const bool output_needs_copyout = output_is_mapped;
                float *d_prefill_output = output_needs_copyout ? impl_->d_C_fp32 : d_output;

                const bool supports_native_small_m =
                    g_rocm_native_quantised_decode_equivalent_scope &&
                    native_quantised_fused && (m >= 2 || countedVerifierRows(m));
                if (supports_native_small_m)
                {
                    if ((k % 32) != 0)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] NativeQuantised fused small-M verifier requires K multiple of 32, got M="
                                  << m << " K=" << k);
                        all_success = false;
                        break;
                    }
                    if (!impl_->d_A_int8 || !impl_->d_scales_A_blockwise || !impl_->d_sums_A_blockwise ||
                        !rocm_kernel->impl_->d_weights_native_quantised ||
                        !rocm_kernel->impl_->d_weights_native_scales ||
                        !rocm_kernel->impl_->d_scatter_partial)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] NativeQuantised fused small-M verifier missing graph-native buffers"
                                  << " A=" << static_cast<const void *>(impl_->d_A_int8)
                                  << " scale_A_blockwise=" << static_cast<const void *>(impl_->d_scales_A_blockwise)
                                  << " sum_A_blockwise=" << static_cast<const void *>(impl_->d_sums_A_blockwise)
                                  << " weights=" << static_cast<const void *>(rocm_kernel->impl_->d_weights_native_quantised)
                                  << " scales=" << static_cast<const void *>(rocm_kernel->impl_->d_weights_native_scales)
                                  << " partial=" << static_cast<const void *>(rocm_kernel->impl_->d_scatter_partial));
                        all_success = false;
                        break;
                    }

                    if (!rocmGemv_native_quantised_small_m_fp32_with_sums_policy(
                            impl_->d_A_int8,
                            rocm_kernel->impl_->d_weights_native_quantised,
                            rocm_kernel->impl_->d_weights_native_scales,
                            rocm_kernel->impl_->d_weights_native_mins,
                            rocm_kernel->impl_->d_weights_native_emins,
                            d_prefill_output,
                            impl_->d_scales_A_blockwise,
                            impl_->d_sums_A_blockwise,
                            rocm_kernel->impl_->d_scatter_partial,
                            m, n, k,
                            rocm_kernel->impl_->native_quantised_codebook_id,
                            requireNativeQuantisedArithmeticPolicyCodebook(
                                rocm_kernel->impl_->native_quantised_codebook_id,
                                rocm_kernel->impl_->native_source_identity),
                            rocm_device_id_, gpu_stream_,
                            VerifierKernelModeScope::rowsFor(m)))
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] NativeQuantised fused small-M verifier GEMV failed for projection "
                                  << i);
                        all_success = false;
                        break;
                    }

                    if (d_bias)
                    {
                        if (!applyFp32EpilogueScoped(
                                d_prefill_output,
                                d_bias,
                                m,
                                n,
                                1.0f,
                                rocm_device_id_,
                                gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] NativeQuantised fused small-M epilogue failed for projection "
                                      << i);
                            all_success = false;
                            break;
                        }
                    }

                    if (output_needs_copyout)
                    {
                        const size_t output_bytes = static_cast<size_t>(m) * n * sizeof(float);
                        const hipError_t copy_err = hipMemcpyAsync(
                            d_output,
                            d_prefill_output,
                            output_bytes,
                            hipMemcpyDeviceToDevice,
                            static_cast<hipStream_t>(gpu_stream_));
                        if (copy_err != hipSuccess)
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Device-alias grouped verifier publication failed: "
                                      << hipGetErrorString(copy_err));
                            all_success = false;
                            break;
                        }
                    }

                if (PerfStatsCollector::isDomainEnabled("kernel"))
                    {
                        PerfStatsCollector::addCounter(
                            "kernel",
                            "rocm_native_quantised_small_m_calls",
                            1.0,
                            "gemm",
                            "rocm:" + std::to_string(rocm_device_id_),
                            PerfStatsCollector::Tags{
                                {"m", std::to_string(m)},
                                {"codebook", std::to_string(static_cast<int>(rocm_kernel->impl_->native_quantised_codebook_id))},
                                {"n", std::to_string(n)},
                                {"k", std::to_string(k)},
                                {"shared_quant", "true"}});

                    }

                    LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                             << " fused native small-M verifier complete");
                    continue;
                }

                if (rocm_kernel->tryPrefillNativeGemm(
                        impl_->d_A_int8,
                        d_prefill_output,
                        impl_->d_scales_A,
                        impl_->d_scales_A_blockwise,
                        d_scales_B,
                        d_bias,
                        m, n, k,
                        1.0f, 0.0f,
                        "ROCmQuantisedGemmKernel::multiply_fused_tensor"))
                {
                    // Copy back from workspace to d_output
                    if (output_needs_copyout)
                    {
                        const size_t output_bytes = static_cast<size_t>(m) * n * sizeof(float);
                        const hipError_t copy_err = hipMemcpyAsync(
                            d_output,
                            d_prefill_output,
                            output_bytes,
                            hipMemcpyDeviceToDevice,
                            static_cast<hipStream_t>(gpu_stream_));
                        if (copy_err != hipSuccess)
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Device-alias prefill publication failed: "
                                      << hipGetErrorString(copy_err));
                            all_success = false;
                            break;
                        }
                    }

                    LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                             << " native prefill complete");
                    continue;
                }

                // Native prefill failed and CK is retired — hard error.
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Projection " << i
                                                                                         << " native-quantised prefill failed (m=" << m << " n=" << n << " k=" << k
                                                                                         << "); CK fallback has been removed.");
                all_success = false;
                break;
            }

            // =========================================================================
            // BATCHED INT8 SCATTER DISPATCH: process all collected VNNI projections
            // in a single 2-kernel launch (1 batched scatter + 1 batched reduce).
            //
            // This replaces N individual kernel launches with 2, saving launch overhead
            // and improving GPU utilization (especially for small-N projections like K/V
            // which individually have only 4 n-blocks but batched with Q get 36).
            // =========================================================================
            if (batch_count > 0 && all_success)
            {
                LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Dispatching BATCHED INT8 SCATTER: "
                          << batch_count << " projections, K=" << k);

                bool batched_ok = false;
                if (fused_uses_blockwise_shared_quant)
                {
                    batched_ok = rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_batched(
                        impl_->d_A_int8,
                        impl_->d_scales_A_blockwise,
                        batch_count,
                        batch_B_ptrs,
                        batch_C_ptrs,
                        batch_scales_B_ptrs,
                        batch_bias_ptrs,
                        batch_N,
                        k,
                        1.0f, 0.0f,
                        rocm_device_id_, gpu_stream_);

                    if (!batched_ok)
                    {
                        batched_ok = rocmGemv_int8_scatter_batched_vnni_blockwise(
                            impl_->d_A_int8,
                            impl_->d_scales_A_blockwise,
                            impl_->d_scatter_partial,
                            batch_count,
                            batch_B_ptrs,
                            batch_C_ptrs,
                            batch_scales_B_ptrs,
                            batch_bias_ptrs,
                            batch_N,
                            k,
                            1.0f, 0.0f,
                            rocm_device_id_, gpu_stream_);
                    }
                }
                else
                {
                    batched_ok = rocmGemv_int8_scatter_batched_vnni(
                        impl_->d_A_int8,
                        impl_->d_scales_A,
                        impl_->d_scatter_partial,
                        batch_count,
                        batch_B_ptrs,
                        batch_C_ptrs,
                        batch_scales_B_ptrs,
                        batch_bias_ptrs,
                        batch_N,
                        k,
                        1.0f, 0.0f,
                        rocm_device_id_, gpu_stream_);
                }

                if (!batched_ok)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Batched INT8 scatter dispatch failed");
                    all_success = false;
                }
                else
                {
                    LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_fused_tensor] Batched INT8 scatter complete: "
                              << batch_count << " projections in 2 kernel launches");
                }
            }

            // Restore original workspace binding
            if (ws && ws != saved_workspace)
            {
                workspace_ = saved_workspace;
            }

            if (all_success)
                publish_projection_outputs();
            return all_success;
        }

    } // namespace rocm
} // namespace llaminar2
