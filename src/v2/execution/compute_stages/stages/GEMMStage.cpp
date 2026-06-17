/**
 * @file GEMMStage.cpp
 * @brief Implementation of GEMMStage
 */

#include "GEMMStage.h"
#include "../ComputeStageUtils.h"
#include "../../../backends/BackendManager.h"
#include "../../../backends/IBackend.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../loaders/PreparedWeightStore.h"

#include <algorithm>
#include <cmath>

namespace llaminar2
{
    namespace
    {
        std::shared_ptr<FP32Tensor> makeScratchFP32(size_t rows, size_t cols, DeviceId device, void *stream)
        {
            auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{rows, cols});
            if (device.is_gpu())
                tensor->allocateOnDevice(device, stream);
            return tensor;
        }

        bool ensureScratchFP32(
            std::shared_ptr<FP32Tensor> &tensor,
            size_t rows,
            size_t cols,
            DeviceId device,
            void *stream,
            const char *name)
        {
            const std::vector<size_t> expected_shape{rows, cols};
            if (!tensor || tensor->shape() != expected_shape)
            {
                if (device.is_gpu() && isGraphCaptureActive())
                {
                    LOG_ERROR("[GEMMStage] Cannot allocate verifier scratch tensor '"
                              << name << "' during graph capture");
                    return false;
                }
                tensor = makeScratchFP32(rows, cols, device, stream);
            }

            if (device.is_gpu() && !tensor->gpu_data_ptr())
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[GEMMStage] Verifier scratch tensor '"
                              << name << "' was not device-resident before graph capture");
                    return false;
                }
                if (!tensor->allocateOnDevice(device, stream))
                {
                    LOG_ERROR("[GEMMStage] Failed to allocate verifier scratch tensor '"
                              << name << "' on " << device.to_string());
                    return false;
                }
            }
            return true;
        }

        bool validateMatrixExtent(
            const TensorBase *tensor,
            const char *tensor_name,
            int required_rows,
            int required_cols)
        {
            if (!tensor)
            {
                LOG_ERROR("[GEMMStage] " << tensor_name << " is not a TensorBase");
                return false;
            }

            const auto &shape = tensor->shape();
            const size_t rows = shape.empty() ? 0 : shape[0];
            const size_t cols = shape.size() < 2 ? 1 : shape[1];
            if (rows < static_cast<size_t>(required_rows) ||
                cols < static_cast<size_t>(required_cols))
            {
                LOG_ERROR("[GEMMStage] Tensor extent mismatch for " << tensor_name
                                                                    << ": required >= ["
                                                                    << required_rows << ", " << required_cols
                                                                    << "], actual=[" << rows << ", " << cols << "]");
                return false;
            }
            return true;
        }

        ITensorGemm *resolvePreparedGemmForStage(
            const char *caller,
            const GEMMStage::Params &params,
            bool is_sliced)
        {
            if (!params.prepared_store || !params.prepared_ref.has_value())
            {
                LOG_ERROR("[" << caller << "] PreparedWeightStore and PreparedWeightRef are required for model GEMM resolution");
                return nullptr;
            }

            ITensorGemm *gemm = nullptr;
            if (is_sliced)
            {
                gemm = params.prepared_store->slicedGemmKernel(
                    params.prepared_ref.value(),
                    params.output_range.start,
                    params.output_range.end);
            }
            else
            {
                gemm = params.prepared_store->gemmKernel(params.prepared_ref.value());
            }

            if (!gemm)
            {
                LOG_ERROR("[" << caller << "] PreparedWeightRef was provided but no GEMM kernel was found in PreparedWeightStore");
            }
            return gemm;
        }

        void markDeviceOutputWritten(TensorBase *tensor, DeviceId device, void *stream)
        {
            if (tensor && device.is_gpu())
            {
                tensor->transitionToWithEvent(TensorCoherenceState::DEVICE_AUTHORITATIVE, device, stream);
            }
        }

        bool copyFP32DeviceRow(
            TensorBase *dst,
            int dst_row,
            int dst_cols,
            const TensorBase *src,
            int src_row,
            int src_cols,
            int copy_cols,
            DeviceId device,
            void *stream,
            const char *label)
        {
            if (!stream)
            {
                LOG_ERROR("[GEMMStage] " << label
                                         << " requires an explicit GPU stream");
                return false;
            }

            IBackend *backend = getBackendFor(device);
            if (!backend)
            {
                LOG_ERROR("[GEMMStage] No backend for " << device.to_string()
                                                        << " while copying " << label);
                return false;
            }

            auto *dst_ptr = static_cast<float *>(dst ? dst->gpu_data_ptr() : nullptr);
            const auto *src_ptr = static_cast<const float *>(src ? src->gpu_data_ptr() : nullptr);
            if (!dst_ptr || !src_ptr)
            {
                LOG_ERROR("[GEMMStage] Null device pointer while copying "
                          << label << " dst=" << static_cast<void *>(dst_ptr)
                          << " src=" << static_cast<const void *>(src_ptr));
                return false;
            }

            const size_t dst_offset = static_cast<size_t>(dst_row) * static_cast<size_t>(dst_cols);
            const size_t src_offset = static_cast<size_t>(src_row) * static_cast<size_t>(src_cols);
            const size_t bytes = static_cast<size_t>(copy_cols) * sizeof(float);
            const bool ok = backend->deviceCopyAsync(
                dst_ptr + dst_offset,
                src_ptr + src_offset,
                bytes,
                device.gpu_ordinal(),
                stream);
            if (!ok)
            {
                LOG_ERROR("[GEMMStage] Device row copy failed for " << label
                                                                    << " bytes=" << bytes);
                return false;
            }
            return true;
        }
    }

    // =============================================================================
    // GEMMStage::Params Implementation
    // =============================================================================

    const float *GEMMStage::Params::getBiasData() const
    {
        // Prefer bias_tensor over raw bias pointer for TensorSlice compatibility
        if (bias_tensor)
        {
            // Cast to TensorBase to access is_fp32_backed()
            auto *bias_base = dynamic_cast<const TensorBase *>(bias_tensor);
            if (!bias_base)
            {
                LOG_WARN("[GEMMStage::Params] bias_tensor is not TensorBase, ignoring.");
                return nullptr;
            }
            // Use unified data access interface - works for FP32Tensor and TensorSlice
            if (bias_base->is_fp32_backed())
            {
                return bias_base->data();
            }
            else
            {
                // Non-FP32-backed bias tensor (e.g., quantized) - not supported
                LOG_WARN("[GEMMStage::Params] bias_tensor is not FP32-backed, ignoring. "
                         "Type: "
                         << bias_base->dtype_name());
                return nullptr;
            }
        }
        // Fall back to raw bias pointer (legacy interface)
        return bias;
    }

    void GEMMStage::Params::validate(const std::string &stage_name) const
    {
        std::ostringstream errors;

        // Check required tensors
        if (!A)
        {
            errors << "Input tensor A is null. ";
        }
        if (!B)
        {
            errors << "Weight tensor B is null. ";
        }
        if (!C)
        {
            errors << "Output tensor C is null. ";
        }

        // Check dimensions
        if (m <= 0 || n <= 0 || k <= 0)
        {
            errors << "Invalid dimensions (m=" << m << ", n=" << n << ", k=" << k << "). ";
        }

        // Check bias requirement
        if (bias_required)
        {
            const float *bias_data = getBiasData();
            if (!bias_data)
            {
                errors << "Bias is required but not provided (neither bias nor bias_tensor set). ";
            }
        }

        // Throw if any errors
        std::string error_str = errors.str();
        if (!error_str.empty())
        {
            throw std::runtime_error("[" + stage_name + "] Configuration error: " + error_str);
        }
    }

    // =============================================================================
    // GEMMStage Implementation
    // =============================================================================

    GEMMStage::GEMMStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool GEMMStage::validatePreparedWeights(std::string *error) const
    {
        if (!params_.B)
            return true;
        if (!params_.prepared_store || !params_.prepared_ref.has_value())
        {
            if (error) *error = "GEMMStage requires PreparedWeightStore and PreparedWeightRef";
            return false;
        }
        if (!params_.prepared_store->contains(params_.prepared_ref.value()))
        {
            if (error) *error = "GEMMStage PreparedWeightRef is not present in PreparedWeightStore";
            return false;
        }
        return true;
    }

    bool GEMMStage::execute(IDeviceContext *ctx)
    {
        ScopedGemmContext gemm_ctx(params_.gemm_context);

        if (!ctx)
        {
            LOG_ERROR("[GEMMStage] Null device context");
            return false;
        }

        // Validate inputs
        if (!params_.A)
        {
            LOG_ERROR("[GEMMStage] Null input A");
            return false;
        }

        if (!params_.B || !params_.C)
        {
            LOG_ERROR("[GEMMStage] Invalid parameters: B=" << params_.B << " C=" << params_.C);
            return false;
        }

        if (params_.m <= 0 || params_.n <= 0 || params_.k <= 0)
        {
            LOG_ERROR("[GEMMStage] Invalid dimensions: m=" << params_.m
                                                           << " n=" << params_.n << " k=" << params_.k);
            return false;
        }

        // === Stage Tracing (Task 3) ===
        traceInput("A", params_.A);
        traceInput("B", params_.B);

        // Check if this is a sliced (tensor-parallel) GEMM
        const bool is_sliced = !params_.output_range.empty();
        const int effective_n = is_sliced ? static_cast<int>(params_.output_range.size()) : params_.n;

        const auto *A_extent = requireTensorBase(params_.A, "input A");
        auto *C_extent = asTensorBase(params_.C, "output C");
        if (!validateMatrixExtent(A_extent, "A", params_.m, params_.k) ||
            !validateMatrixExtent(C_extent, "C", params_.m, effective_n))
        {
            return false;
        }
        if (params_.gate_input)
        {
            const auto *gate_extent = requireTensorBase(params_.gate_input, "gate input");
            if (!validateMatrixExtent(gate_extent, "gate_input", params_.m, params_.k))
                return false;
        }

        LOG_DEBUG("[GEMMStage] Execute GEMM: " << params_.m << "x" << effective_n << "x" << params_.k
                                               << (is_sliced ? " (SLICED)" : "")
                                               << " weight ptr=" << static_cast<const void *>(params_.B)
                                               << " weight shape=[" << (params_.B ? params_.B->shape()[0] : 0) << ","
                                               << (params_.B ? params_.B->shape()[1] : 0) << "]"
                                               << " input ptr=" << static_cast<const void *>(params_.A)
                                               << " input_type=" << params_.A->dtype_name()
                                               << " output_type=" << params_.C->dtype_name());

        // Debug: Log input values for Wo projection (weight shape [896, 448] or [896, 896])
        // Only log if input is FP32 (Q8_1 data() throws to prevent accidental dequantization)
        // NOTE: ONLY call data() when TRACE logging is actually enabled - it triggers D2H transfer!
#if 0 // DISABLED: calling data() triggers expensive D2H transfer even if LOG_TRACE doesn't print
        if (params_.B && params_.B->shape()[0] == 896 && (params_.B->shape()[1] == 448 || params_.B->shape()[1] == 896))
        {
            if (std::string(params_.A->dtype_name()) == "FP32")
            {
                const float *input_data = params_.A->data();
                if (input_data)
                {
                    LOG_TRACE("[GEMMStage] Wo input[0:8]=" << std::setprecision(10)
                                                           << input_data[0] << "," << input_data[1] << "," << input_data[2] << "," << input_data[3] << ","
                                                           << input_data[4] << "," << input_data[5] << "," << input_data[6] << "," << input_data[7]
                                                           << " weight_k=" << params_.B->shape()[1]);
                }
            }
        }
#endif

        // Cast weights to TensorBase for diagnostics and tensor-aware kernel calls.
        auto *B_base = requireTensorBase(params_.B, "weight B");

        // Get kernel — use stage-level cache to avoid store lookup per call.
        llaminar2::ITensorGemm *gemm = nullptr;

        if (cache_resolved_)
        {
            // Fast path: reuse previously-resolved kernel (no mutex)
            gemm = cached_gemm_;
        }
        else
        {
            gemm = resolvePreparedGemmForStage("GEMMStage", params_, is_sliced);
            if (!gemm)
                return false;
            cached_gemm_ = gemm;
            cache_resolved_ = true;
            if (is_sliced)
            {
                LOG_DEBUG("[GEMMStage] Using prepared sliced kernel for rows [" << params_.output_range.start
                                                                                << ", " << params_.output_range.end << ")");
            }
        }

        if (!gemm)
        {
            LOG_ERROR("[GEMMStage] Failed to get GEMM kernel for weight tensor");
            return false;
        }

        // Thread GPU stream for graph capture
        gemm->setGPUStream(gpuStream());

        LOG_DEBUG("[GEMMStage] Got kernel ptr=" << static_cast<const void *>(gemm)
                                                << " for weight ITensor*=" << static_cast<const void *>(params_.B)
                                                << " TensorBase*=" << static_cast<const void *>(B_base));

        if (params_.force_decode_equivalent_verifier_prefill && params_.m > 1)
        {
            auto *A_base = requireTensorBase(params_.A, "input A");
            auto *C_base = asTensorBase(params_.C, "output C");
            return executeDecodeEquivalentVerifierPrefill(
                A_base, C_base, gemm, effective_n);
        }

        // Fused SwiGLU + GEMM: output = W @ (silu(gate) * up)
        // Both CPU and GPU kernels implement multiply_tensor_with_fused_swiglu().
        if (params_.gate_input)
        {
            auto *gate_base = requireTensorBase(params_.gate_input, "gate input");
            auto *A_base_up = requireTensorBase(params_.A, "input A (up)");
            auto *C_base = asTensorBase(params_.C, "output C");

            if (gemm->multiply_tensor_with_fused_swiglu(
                    gate_base, A_base_up, C_base,
                    params_.m, effective_n, params_.k,
                    params_.alpha, params_.beta,
                    getWorkspace()))
            {
                markDeviceOutputWritten(C_base, params_.device_id, gpuStream());
                LOG_DEBUG("[GEMMStage] Fused SwiGLU+GEMM completed via ITensorGemm");
                traceOutput("C", params_.C);
                return true;
            }

            LOG_DEBUG("[GEMMStage] Fused SwiGLU+GEMM unavailable; falling back to separate SwiGLU + GEMM");
            auto *swiglu_output = const_cast<TensorBase *>(A_base_up);
            auto *activation = dynamic_cast<IActivationTensor *>(swiglu_output);
            if (!activation)
            {
                LOG_ERROR("[GEMMStage] Cannot run SwiGLU fallback: up tensor is not an activation tensor");
                return false;
            }

            auto swiglu = activation->createSwiGLU();
            if (!swiglu)
            {
                LOG_ERROR("[GEMMStage] Cannot run SwiGLU fallback: failed to create SwiGLU kernel");
                return false;
            }
            swiglu->setGPUStream(gpuStream());

            if (!swiglu->apply_tensor(
                    gate_base, A_base_up, swiglu_output,
                    params_.m, params_.k,
                    /*add_residual=*/false,
                    params_.mpi_ctx,
                    params_.device_id.toKernelDeviceIndex()))
            {
                LOG_ERROR("[GEMMStage] SwiGLU fallback activation failed");
                return false;
            }
            markDeviceOutputWritten(swiglu_output, params_.device_id, gpuStream());

            bool success = gemm->multiply_tensor(
                swiglu_output, C_base,
                params_.m, effective_n, params_.k,
                params_.transpose_B,
                params_.alpha, params_.beta,
                nullptr, // bias
                params_.mpi_ctx, params_.device_id.toKernelDeviceIndex(),
                getWorkspace());
            if (success)
            {
                markDeviceOutputWritten(C_base, params_.device_id, gpuStream());
                traceOutput("C", params_.C);
            }
            return success;
        }

        // Primary path: use tensor-aware multiply_tensor for type-aware dispatch.
        // Works for CPU (NativeVNNI, FloatingPointGemm) and GPU (CUDA/ROCm) kernels.
        {
            auto *A_base = requireTensorBase(params_.A, "input A");
            auto *C_base = asTensorBase(params_.C, "output C");

            LOG_DEBUG("[GEMMStage] Using multiply_tensor for type-aware dispatch: "
                      << "input_type=" << params_.A->dtype_name()
                      << " output_type=" << params_.C->dtype_name());
            bool success = gemm->multiply_tensor(
                A_base, C_base,
                params_.m, effective_n, params_.k,
                params_.transpose_B,
                params_.alpha, params_.beta,
                nullptr, // bias
                params_.mpi_ctx, params_.device_id.toKernelDeviceIndex(),
                getWorkspace());

            if (success)
            {
                markDeviceOutputWritten(C_base, params_.device_id, gpuStream());
                traceOutput("C", params_.C);
            }
            return success;
        }
    }

    bool GEMMStage::executeDecodeEquivalentVerifierPrefill(
        const TensorBase *A_base,
        TensorBase *C_base,
        ITensorGemm *gemm,
        int effective_n)
    {
        if (!A_base || !C_base || !gemm)
            return false;
        if (params_.m > 4)
        {
            LOG_ERROR("[GEMMStage] Decode-equivalent verifier prefill is only supported "
                      << "for tiny MTP verifier batches, got m=" << params_.m);
            return false;
        }

        const bool is_gpu = params_.device_id.is_gpu();
        void *stream = gpuStream();
        if (is_gpu && !stream)
        {
            LOG_ERROR("[GEMMStage] Decode-equivalent GPU verifier prefill requires an explicit stream");
            return false;
        }

        if (!ensureScratchFP32(verifier_input_row_, 1, static_cast<size_t>(params_.k),
                               params_.device_id, stream, "input_row") ||
            !ensureScratchFP32(verifier_output_row_, 1, static_cast<size_t>(effective_n),
                               params_.device_id, stream, "output_row"))
        {
            return false;
        }
        if (params_.gate_input &&
            !ensureScratchFP32(verifier_gate_row_, 1, static_cast<size_t>(params_.k),
                               params_.device_id, stream, "gate_row"))
        {
            return false;
        }

        const float *input_data = is_gpu ? nullptr : A_base->data();
        const auto *gate_base = params_.gate_input
                                    ? requireTensorBase(params_.gate_input, "gate input")
                                    : nullptr;
        const float *gate_data = (!is_gpu && gate_base) ? gate_base->data() : nullptr;
        float *output_data = is_gpu ? nullptr : C_base->mutable_data();
        if (!is_gpu && (!input_data || !output_data))
        {
            LOG_ERROR("[GEMMStage] Decode-equivalent verifier prefill requires FP32 host-visible tensors");
            return false;
        }
        if (params_.gate_input && !gate_base)
            return false;
        if (!is_gpu && params_.gate_input && !gate_data)
        {
            LOG_ERROR("[GEMMStage] Decode-equivalent fused SwiGLU verifier prefill requires host-visible gate tensor");
            return false;
        }

        if (is_gpu)
        {
            if (!A_base->gpu_data_ptr())
            {
                LOG_ERROR("[GEMMStage] Decode-equivalent verifier input is not device-resident on "
                          << params_.device_id.to_string()
                          << "; StageBufferContract/BufferArena coherence must prepare graph inputs");
                return false;
            }
            if (params_.gate_input && !gate_base->gpu_data_ptr())
            {
                LOG_ERROR("[GEMMStage] Decode-equivalent verifier gate input is not device-resident on "
                          << params_.device_id.to_string()
                          << "; StageBufferContract/BufferArena coherence must prepare graph inputs");
                return false;
            }
            if (!C_base->allocateOnDevice(params_.device_id, stream))
            {
                LOG_ERROR("[GEMMStage] Failed to prepare verifier output tensor on "
                          << params_.device_id.to_string());
                return false;
            }
        }

        bool success = true;
        for (int row = 0; row < params_.m; ++row)
        {
            if (is_gpu)
            {
                if (!copyFP32DeviceRow(
                        verifier_input_row_.get(), 0, params_.k,
                        A_base, row, params_.k,
                        params_.k, params_.device_id, stream,
                        "verifier_input_row"))
                {
                    success = false;
                    break;
                }
                if (params_.gate_input &&
                    !copyFP32DeviceRow(
                        verifier_gate_row_.get(), 0, params_.k,
                        gate_base, row, params_.k,
                        params_.k, params_.device_id, stream,
                        "verifier_gate_row"))
                {
                    success = false;
                    break;
                }
                markDeviceOutputWritten(verifier_input_row_.get(), params_.device_id, stream);
                if (params_.gate_input)
                    markDeviceOutputWritten(verifier_gate_row_.get(), params_.device_id, stream);

                if (params_.beta != 0.0f &&
                    !copyFP32DeviceRow(
                        verifier_output_row_.get(), 0, effective_n,
                        C_base, row, effective_n,
                        effective_n, params_.device_id, stream,
                        "verifier_existing_output_row"))
                {
                    success = false;
                    break;
                }
            }
            else
            {
                std::copy(input_data + static_cast<size_t>(row) * params_.k,
                          input_data + static_cast<size_t>(row + 1) * params_.k,
                          verifier_input_row_->mutable_data());
                if (params_.gate_input)
                {
                    std::copy(gate_data + static_cast<size_t>(row) * params_.k,
                              gate_data + static_cast<size_t>(row + 1) * params_.k,
                              verifier_gate_row_->mutable_data());
                }
                if (params_.beta != 0.0f)
                {
                    std::copy(output_data + static_cast<size_t>(row) * effective_n,
                              output_data + static_cast<size_t>(row + 1) * effective_n,
                              verifier_output_row_->mutable_data());
                }
            }

            const bool row_success = params_.gate_input
                                         ? gemm->multiply_tensor_with_fused_swiglu(
                                               verifier_gate_row_.get(),
                                               verifier_input_row_.get(),
                                               verifier_output_row_.get(),
                                               1,
                                               effective_n,
                                               params_.k,
                                               params_.alpha,
                                               params_.beta,
                                               getWorkspace())
                                         : gemm->multiply_tensor(
                                               verifier_input_row_.get(),
                                               verifier_output_row_.get(),
                                               1,
                                               effective_n,
                                               params_.k,
                                               params_.transpose_B,
                                               params_.alpha,
                                               params_.beta,
                                               nullptr,
                                               params_.mpi_ctx,
                                               params_.device_id.toKernelDeviceIndex(),
                                               getWorkspace());
            if (!row_success)
            {
                LOG_ERROR("[GEMMStage] Decode-equivalent verifier row "
                          << row << " failed");
                success = false;
                break;
            }

            /*
             * The row GEMV writes verifier_output_row_ on the stage stream.
             * Publish that scratch explicitly before the following device row
             * copy consumes it. Without this handoff, later coherence-aware
             * code can legally observe the previous host shadow while the
             * device bytes are already correct.
             */
            if (is_gpu)
                markDeviceOutputWritten(
                    verifier_output_row_.get(),
                    params_.device_id,
                    stream);

            if (is_gpu)
            {
                if (!copyFP32DeviceRow(
                        C_base, row, effective_n,
                        verifier_output_row_.get(), 0, effective_n,
                        effective_n, params_.device_id, stream,
                        "verifier_output_row"))
                {
                    success = false;
                    break;
                }
            }
            else
            {
                std::copy(verifier_output_row_->data(),
                          verifier_output_row_->data() + effective_n,
                          output_data + static_cast<size_t>(row) * effective_n);
            }
        }

        if (success)
        {
            if (is_gpu)
                markDeviceOutputWritten(C_base, params_.device_id, stream);
            PerfStatsCollector::addCounter(
                "mtp",
                "gemm_decode_equivalent_verifier_prefill_rows",
                static_cast<double>(params_.m),
                {},
                params_.device_id.to_string(),
                {{"stage", "GEMM"}});
            traceOutput("C", params_.C);
        }
        return success;
    }

    size_t GEMMStage::estimatedFlops() const
    {
        // GEMM: 2 * M * N * K (multiply + add)
        return static_cast<size_t>(2) * params_.m * params_.n * params_.k;
    }

    size_t GEMMStage::estimatedMemoryBytes() const
    {
        // A: m * k reads, B: k * n reads, C: m * n writes (+ reads if beta != 0)
        size_t a_bytes = static_cast<size_t>(params_.m) * params_.k * sizeof(float);
        size_t c_bytes = static_cast<size_t>(params_.m) * params_.n * sizeof(float);

        // B may be quantized, so we estimate based on tensor
        // For now, assume FP32 - tensor introspection would be better
        size_t b_bytes = static_cast<size_t>(params_.k) * params_.n * sizeof(float);

        return a_bytes + b_bytes + c_bytes;
    }

    bool GEMMStage::supportsBackend(ComputeBackendType backend) const
    {
        // Unified GEMMStage supports all backends via KernelFactory dispatch
        // KernelFactory will select the appropriate kernel based on tensor device affinity
        switch (backend)
        {
        case ComputeBackendType::CPU:
            return true;
        case ComputeBackendType::GPU_CUDA:
#ifdef HAVE_CUDA
            return true; // KernelFactory can create CUDA kernels
#else
            return false;
#endif
        case ComputeBackendType::GPU_ROCM:
#ifdef HAVE_ROCM
            return true; // KernelFactory can create ROCm kernels
#else
            return false;
#endif
        case ComputeBackendType::GPU_VULKAN:
        case ComputeBackendType::GPU_METAL:
            return false; // Not yet implemented
        default:
            return false;
        }
    }

    StageDumpInfo GEMMStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        // Input A: activation tensor
        if (params_.A)
        {
            info.addInput("A", params_.A, params_.m, params_.k);
        }

        // Weight tensor B
        info.addWeight("B", params_.B);

        // Output C
        if (params_.C)
        {
            info.addOutput("C", params_.C, params_.m, params_.n);
        }

        // Optional inputs - use unified interface for bias
        const float *bias_data = params_.getBiasData();
        if (bias_data)
        {
            info.addInput("bias", bias_data, 1, params_.n);
        }
        if (params_.gate_input)
        {
            info.addInput("gate_input", params_.gate_input, params_.m, params_.n);
        }

        // Scalar params
        info.addScalarInt("m", params_.m);
        info.addScalarInt("n", params_.n);
        info.addScalarInt("k", params_.k);
        info.addScalar("alpha", params_.alpha);
        info.addScalar("beta", params_.beta);
        info.addScalarBool("transpose_B", params_.transpose_B);
        info.addScalarBool("do_swiglu", params_.do_swiglu);
        info.addScalarInt("device_id", params_.device_id.toKernelDeviceIndex());

        return info;
    }

    StageBufferRequirements GEMMStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (!params_.A || !params_.B || !params_.C)
            return reqs; // Empty if tensors not set

        // Convert tensor type to buffer tensor type
        BufferTensorType a_type = toBufferTensorType(params_.A->native_type());
        BufferTensorType b_type = toBufferTensorType(params_.B->native_type());
        BufferTensorType c_type = toBufferTensorType(params_.C->native_type());

        // INPUT buffer (activations)
        reqs.addInput("A", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.k)}, a_type);

        // WEIGHT buffer (read-only, may be quantized)
        reqs.addWeight("B", {static_cast<size_t>(params_.k), static_cast<size_t>(params_.n)}, b_type);

        // OUTPUT buffer
        reqs.addOutput("C", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.n)}, c_type);

        // Optional bias (check both raw pointer and tensor)
        const float *bias_data = params_.getBiasData();
        if (bias_data)
        {
            reqs.addWeight("bias", {static_cast<size_t>(params_.n)}, BufferTensorType::FP32);
        }

        // Optional gate_input for SwiGLU fusion
        if (params_.gate_input)
        {
            BufferTensorType gate_type = toBufferTensorType(params_.gate_input->native_type());
            reqs.addInput("gate_input", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.k)}, gate_type);
        }

        return reqs;
    }

    // =============================================================================
    // IWorkspaceConsumerStage Implementation
    // =============================================================================

    IWorkspaceConsumer *GEMMStage::getKernelAsWorkspaceConsumer()
    {
        if (!params_.B)
        {
            LOG_WARN("[GEMMStage::getKernelAsWorkspaceConsumer] Weight tensor B not set");
            return nullptr;
        }

        auto *B_base = dynamic_cast<TensorBase *>(const_cast<ITensor *>(params_.B));
        if (!B_base)
        {
            LOG_WARN("[GEMMStage::getKernelAsWorkspaceConsumer] Weight tensor B is not TensorBase");
            return nullptr;
        }

        const bool is_sliced = !params_.output_range.empty();
        ITensorGemm *gemm = resolvePreparedGemmForStage(
            "GEMMStage::getKernelAsWorkspaceConsumer", params_, is_sliced);

        return dynamic_cast<IWorkspaceConsumer *>(gemm);
    }

    StageBufferContract GEMMStage::bufferContract() const
    {
        if (!params_.a_buffer_id || !params_.c_buffer_id)
            return {};

        auto contract = StageBufferContract::build()
            .addInput(*params_.a_buffer_id);
        if (params_.gate_input && params_.gate_buffer_id)
            contract.addInput(*params_.gate_buffer_id);
        contract.addOutput(*params_.c_buffer_id);
        // Model weight B is not arena-managed
        if (params_.B)
            contract.addWeight(const_cast<ITensor *>(params_.B));
        if (params_.bias_tensor)
            contract.addWeight(const_cast<ITensor *>(static_cast<const ITensor *>(params_.bias_tensor)));
        return contract;
    }

} // namespace llaminar2
