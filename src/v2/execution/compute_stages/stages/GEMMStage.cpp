/**
 * @file GEMMStage.cpp
 * @brief Implementation of GEMMStage
 */

#include "GEMMStage.h"
#include "../ComputeStageUtils.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../kernels/cpu/gemm_v4/QuantisedGemmKernel.h"
#include "../../../interfaces/IWorkspaceConsumer.h"

namespace llaminar2
{

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

    bool GEMMStage::execute(IDeviceContext *ctx)
    {
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

        // Cast weights to TensorBase for KernelFactory
        auto *B_base = requireTensorBase(params_.B, "weight B");

        // Get kernel from KernelFactory (which caches by tensor + device)
        auto target_dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(params_.device_id);
        llaminar2::ITensorGemm *gemm = nullptr;

        if (is_sliced)
        {
            // Use sliced kernel for tensor parallelism
            gemm = llaminar::v2::kernels::KernelFactory::getOrCreateGemmSliced(
                B_base, params_.output_range.start, params_.output_range.end);
            LOG_DEBUG("[GEMMStage] Using sliced kernel for rows [" << params_.output_range.start
                                                                   << ", " << params_.output_range.end << ")");
        }
        else
        {
            // Use device-targeted kernel creation
            gemm = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(B_base, params_.device_id);
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

        // Cast to QuantisedGemmKernel for full API access
        auto *qgemm = dynamic_cast<gemm_v4::QuantisedGemmKernel *>(gemm);

        // Dimension validation
        if (qgemm)
        {
            int kernel_n = qgemm->get_n();
            int kernel_k = qgemm->get_k();
            LOG_DEBUG("[GEMMStage] Kernel dimensions: N=" << kernel_n << " K=" << kernel_k
                                                          << ", params: n=" << effective_n << " k=" << params_.k);
            if (kernel_n != effective_n || kernel_k != params_.k)
            {
                LOG_ERROR("[GEMMStage] DIMENSION MISMATCH! kernel N=" << kernel_n << " vs params n=" << effective_n
                                                                      << ", kernel K=" << kernel_k << " vs params k=" << params_.k);
                return false;
            }

            // Check if we have a gate input for SwiGLU fusion
            const float *gate_fp32 = params_.gate_input ? params_.gate_input->fp32_data() : nullptr;

            // If no SwiGLU fusion, use multiply_tensor for type-aware dispatch.
            // This handles all input/output type combinations efficiently:
            // - Q8_1→Q8_1: Direct JIT kernel (zero-copy)
            // - Q8_1→FP32: multiply_q8_1_direct (JIT kernel, no dequant buffer)
            // - FP32→Q8_1: multiply_to_q8_1 (quantize once, JIT kernel)
            // - FP32→FP32: Standard GEMM
            //
            // IMPORTANT: Don't use multiply_fused for Q8_1 input without gate fusion!
            // multiply_fused calls fp32_data() which dequantizes the entire Q8_1 tensor
            // into a buffer EVERY CALL - this was causing 25x slowdown for down_proj.
            if (!gate_fp32)
            {
                // Cast input/output to TensorBase for multiply_tensor
                auto *A_base = requireTensorBase(params_.A, "input A");
                auto *C_base = asTensorBase(params_.C, "output C");

                LOG_DEBUG("[GEMMStage] Using multiply_tensor for type-aware dispatch: "
                          << "input_type=" << params_.A->dtype_name()
                          << " output_type=" << params_.C->dtype_name());
                bool success = qgemm->multiply_tensor(
                    A_base, C_base,
                    params_.m, effective_n, params_.k,
                    params_.transpose_B,
                    params_.alpha, params_.beta,
                    nullptr, // bias (not supported in GEMMStage yet)
                    params_.mpi_ctx, params_.device_id.toKernelDeviceIndex(),
                    getWorkspace()); // Pass workspace from IWorkspaceConsumerStage

                // === Stage Tracing (Task 3) ===
                if (success)
                    traceOutput("C", params_.C);
                return success;
            }

            // For SwiGLU fusion, use multiply_fused which requires FP32 pointers.
            // Note: If gate_input is Q8_1, gate_fp32 calls fp32_data() which dequantizes.
            // This is acceptable because gate fusion is compute-bound, not memory-bound.
            float *output_fp32 = params_.C->mutable_data();
            const float *input_fp32 = params_.A->fp32_data();

            if (!input_fp32 || !output_fp32)
            {
                LOG_ERROR("[GEMMStage] Failed to get FP32 data from tensors");
                return false;
            }

            // Use getBiasData() for TensorSlice compatibility (TP sharded bias)
            const float *bias_data = params_.getBiasData();

            bool success = qgemm->multiply_fused(
                input_fp32,
                output_fp32,
                params_.m, effective_n, params_.k,
                bias_data,              // Fused bias (via unified interface)
                nullptr,                // No attention mask
                false,                  // No softmax
                nullptr, nullptr,       // No softmax buffers
                (params_.beta != 0.0f), // Accumulate if beta != 0
                params_.alpha, params_.beta,
                params_.mpi_ctx, params_.device_id.toKernelDeviceIndex(),
                gate_fp32,
                params_.do_swiglu); // CPU multiply_fused doesn't have workspace param

            // === Stage Tracing (Task 3) ===
            if (success)
                traceOutput("C", params_.C);
            return success;
        }

        // Cast A/C for FP32 GEMM fallback
        auto *A_base_fallback = requireTensorBase(params_.A, "input A fallback");
        auto *C_base_fallback = asTensorBase(params_.C, "output C fallback");

        // FP32 GEMM fallback (for non-quantized weights)
        // Use multiply_tensor if available for type-aware dispatch
        if (gemm->multiply_tensor(A_base_fallback, C_base_fallback, params_.m, effective_n, params_.k,
                                  params_.transpose_B, params_.alpha, params_.beta,
                                  nullptr, // bias (not supported in GEMMStage yet)
                                  params_.mpi_ctx, params_.device_id.toKernelDeviceIndex(),
                                  getWorkspace())) // Pass workspace from IWorkspaceConsumerStage
        {
            // === Stage Tracing (Task 3) ===
            traceOutput("C", params_.C);
            return true;
        }

        // Final fallback: extract FP32 data manually
        const float *input_fp32 = params_.A->fp32_data();
        float *output_fp32 = params_.C->mutable_data();

        if (!input_fp32 || !output_fp32)
        {
            LOG_ERROR("[GEMMStage] Failed to get FP32 data from tensors");
            return false;
        }

        bool success = gemm->multiply(
            input_fp32,
            output_fp32,
            params_.m, effective_n, params_.k,
            params_.transpose_B,
            params_.alpha, params_.beta,
            params_.mpi_ctx, params_.device_id.toKernelDeviceIndex(),
            getWorkspace()); // Pass workspace from IWorkspaceConsumerStage

        // === Stage Tracing (Task 3) ===
        if (success)
            traceOutput("C", params_.C);
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
            reqs.addInput("gate_input", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.n)}, gate_type);
        }

        return reqs;
    }

    // =============================================================================
    // IWorkspaceConsumerStage Implementation
    // =============================================================================

    IWorkspaceConsumer *GEMMStage::getKernelAsWorkspaceConsumer()
    {
        // Get kernel from KernelFactory (which caches by tensor + device)
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

        auto target_dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(params_.device_id);
        ITensorGemm *gemm = nullptr;

        if (!params_.output_range.empty())
        {
            // Sliced kernel for tensor parallelism
            gemm = llaminar::v2::kernels::KernelFactory::getOrCreateGemmSliced(
                B_base, params_.output_range.start, params_.output_range.end);
        }
        else
        {
            // Standard GEMM kernel
            gemm = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(B_base, params_.device_id);
        }

        return dynamic_cast<IWorkspaceConsumer *>(gemm);
    }

} // namespace llaminar2
