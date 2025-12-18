/**
 * @file ComputeStage.cpp
 * @brief Implementation of compute stage abstractions
 * @author David Sanftenberg
 * @date December 2025
 */

#include "ComputeStage.h"
#include "../utils/Logger.h"
#include "../utils/DebugEnv.h"
#include "../utils/OpenMPUtils.h"
#include "../kernels/KernelFactory.h"
#include "../kernels/cpu/gemm_v4/QuantisedGemmKernel.h"
#include "../kernels/cpu/gemm_v4/FusedGEMM.h"
#include "../tensors/UnifiedKVCache.h"
#include "../tensors/SIMDHelpers.h"
#include "../tensors/Tensors.h" // For TensorBase::rows(), cols(), dtype_name()
#include "../pipelines/attention/MpiAttentionOrchestrator.h"
#include "../pipelines/MPIStrategy.h"
#include "../utils/MPIContext.h"

#include <cstring>
#include <cmath>
#include <sstream>
#include <limits>
#include <omp.h>

// Note: In production, GEMM would delegate to KernelFactory
// For now, we use placeholder implementation to avoid cross-namespace dependencies

#include <mpi.h>

namespace llaminar2
{

    // =============================================================================
    // Stage Type Names
    // =============================================================================

    const char *computeStageTypeName(ComputeStageType type)
    {
        switch (type)
        {
        case ComputeStageType::GEMM:
            return "GEMM";
        case ComputeStageType::GEMM_BIAS:
            return "GEMM_BIAS";
        case ComputeStageType::GEMM_FUSED_QKV:
            return "GEMM_FUSED_QKV";
        case ComputeStageType::GEMM_FUSED_GATE_UP:
            return "GEMM_FUSED_GATE_UP";
        case ComputeStageType::RMS_NORM:
            return "RMS_NORM";
        case ComputeStageType::LAYER_NORM:
            return "LAYER_NORM";
        case ComputeStageType::SWIGLU:
            return "SWIGLU";
        case ComputeStageType::GELU:
            return "GELU";
        case ComputeStageType::SILU:
            return "SILU";
        case ComputeStageType::ROPE:
            return "ROPE";
        case ComputeStageType::ATTENTION:
            return "ATTENTION";
        case ComputeStageType::ATTENTION_QK:
            return "ATTENTION_QK";
        case ComputeStageType::ATTENTION_SOFTMAX:
            return "ATTENTION_SOFTMAX";
        case ComputeStageType::ATTENTION_V:
            return "ATTENTION_V";
        case ComputeStageType::ADD_RESIDUAL:
            return "ADD_RESIDUAL";
        case ComputeStageType::SCALE:
            return "SCALE";
        case ComputeStageType::MOE_ROUTER:
            return "MOE_ROUTER";
        case ComputeStageType::MOE_EXPERT_FFN:
            return "MOE_EXPERT_FFN";
        case ComputeStageType::MOE_COMBINE:
            return "MOE_COMBINE";
        case ComputeStageType::ALLREDUCE:
            return "ALLREDUCE";
        case ComputeStageType::ALLGATHER:
            return "ALLGATHER";
        case ComputeStageType::COPY:
            return "COPY";
        case ComputeStageType::DEQUANTIZE:
            return "DEQUANTIZE";
        case ComputeStageType::QUANTIZE:
            return "QUANTIZE";
        case ComputeStageType::EMBEDDING:
            return "EMBEDDING";
        case ComputeStageType::LM_HEAD:
            return "LM_HEAD";
        case ComputeStageType::FINAL_NORM:
            return "FINAL_NORM";
        default:
            return "UNKNOWN";
        }
    }

    // =============================================================================
    // Helper: Safe FP32 data access for getDumpInfo()
    // For Q8_1 tensors, uses fp32_data() (explicit dequant for debugging only)
    // For other tensors, uses data()
    // =============================================================================

    static const float *getSafeFp32Data(const TensorBase *tensor)
    {
        if (!tensor)
            return nullptr;

        if (tensor->native_type() == TensorType::Q8_1)
        {
            // Q8_1 tensors throw on data() - use explicit fp32_data() for debugging
            auto *q8_1 = dynamic_cast<const Q8_1Tensor *>(tensor);
            return q8_1 ? q8_1->fp32_data() : nullptr;
        }

        return tensor->data();
    }

    // =============================================================================
    // StageDumpInfo Implementation
    // =============================================================================

    StageDumpInfo &StageDumpInfo::addWeight(const char *name, const TensorBase *tensor)
    {
        if (tensor)
        {
            weights.push_back({name, tensor, nullptr, 0,
                               tensor->rows(), tensor->cols(),
                               tensor->dtype_name()});
        }
        return *this;
    }

    StageDumpInfo &StageDumpInfo::addInput(const char *name, const TensorBase *tensor, size_t rows, size_t cols)
    {
        // Extract FP32 data from tensor for dumping - works for all tensor types
        const float *data = tensor ? tensor->fp32_data() : nullptr;
        const char *dtype = tensor ? tensor->dtype_name() : "FP32";
        inputs.push_back({name, data, rows, cols, dtype, sizeof(float)});
        return *this;
    }

    StageDumpInfo &StageDumpInfo::addOutput(const char *name, const TensorBase *tensor, size_t rows, size_t cols)
    {
        // Extract FP32 data from tensor for dumping - works for all tensor types
        const float *data = tensor ? tensor->fp32_data() : nullptr;
        const char *dtype = tensor ? tensor->dtype_name() : "FP32";
        outputs.push_back({name, data, rows, cols, dtype, sizeof(float)});
        return *this;
    }

    // =============================================================================
    // GEMMStage Implementation
    // =============================================================================

    GEMMStage::GEMMStage(Params params) : params_(std::move(params)) {}

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

        LOG_DEBUG("[GEMMStage] Execute GEMM: " << params_.m << "x" << params_.n << "x" << params_.k
                                               << " weight ptr=" << static_cast<const void *>(params_.B)
                                               << " weight shape=[" << (params_.B ? params_.B->shape()[0] : 0) << ","
                                               << (params_.B ? params_.B->shape()[1] : 0) << "]"
                                               << " input_type=" << params_.A->dtype_name()
                                               << " output_type=" << params_.C->dtype_name());

        // Get cached kernel from KernelFactory (handles weight packing once)
        auto *gemm = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(params_.B);
        if (!gemm)
        {
            LOG_ERROR("[GEMMStage] Failed to get GEMM kernel for weight tensor");
            return false;
        }

        LOG_DEBUG("[GEMMStage] Got kernel ptr=" << static_cast<const void *>(gemm)
                                                << " for weight ptr=" << static_cast<const void *>(params_.B));

        // Cast to QuantisedGemmKernel for full API access
        auto *qgemm = dynamic_cast<gemm_v4::QuantisedGemmKernel *>(gemm);

        // Dimension validation
        if (qgemm)
        {
            int kernel_n = qgemm->get_n();
            int kernel_k = qgemm->get_k();
            LOG_DEBUG("[GEMMStage] Kernel dimensions: N=" << kernel_n << " K=" << kernel_k
                                                          << ", params: n=" << params_.n << " k=" << params_.k);
            if (kernel_n != params_.n || kernel_k != params_.k)
            {
                LOG_ERROR("[GEMMStage] DIMENSION MISMATCH! kernel N=" << kernel_n << " vs params n=" << params_.n
                                                                      << ", kernel K=" << kernel_k << " vs params k=" << params_.k);
                return false;
            }

            // Check if we have a gate input for SwiGLU fusion
            const float *gate_fp32 = params_.gate_input ? params_.gate_input->fp32_data() : nullptr;

            // If output is Q8_1 and no fusion needed, use multiply_tensor for type-aware dispatch
            // This handles: FP32→Q8_1, Q8_1→Q8_1, Q8_1→FP32, FP32→FP32 internally
            if (!gate_fp32 && params_.C->native_type() == TensorType::Q8_1)
            {
                LOG_DEBUG("[GEMMStage] Using multiply_tensor for Q8_1 output type dispatch");
                return qgemm->multiply_tensor(
                    params_.A, params_.C,
                    params_.m, params_.n, params_.k,
                    params_.transpose_B,
                    params_.alpha, params_.beta,
                    params_.mpi_ctx, params_.device_idx);
            }

            // For SwiGLU fusion or FP32 output, use multiply_fused
            // (multiply_fused requires FP32 output for gate fusion)
            float *output_fp32 = params_.C->mutable_data();
            const float *input_fp32 = params_.A->fp32_data();

            if (!input_fp32 || !output_fp32)
            {
                LOG_ERROR("[GEMMStage] Failed to get FP32 data from tensors");
                return false;
            }

            return qgemm->multiply_fused(
                input_fp32,
                output_fp32,
                params_.m, params_.n, params_.k,
                params_.bias,           // Fused bias
                nullptr,                // No attention mask
                false,                  // No softmax
                nullptr, nullptr,       // No softmax buffers
                (params_.beta != 0.0f), // Accumulate if beta != 0
                params_.alpha, params_.beta,
                params_.mpi_ctx, params_.device_idx,
                gate_fp32,
                params_.do_swiglu);
        }

        // FP32 GEMM fallback (for non-quantized weights)
        // Use multiply_tensor if available for type-aware dispatch
        if (gemm->multiply_tensor(params_.A, params_.C, params_.m, params_.n, params_.k,
                                  params_.transpose_B, params_.alpha, params_.beta,
                                  params_.mpi_ctx, params_.device_idx))
        {
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

        return gemm->multiply(
            input_fp32,
            output_fp32,
            params_.m, params_.n, params_.k,
            params_.alpha, params_.beta);
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

    StageDumpInfo GEMMStage::getDumpInfo() const
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

        // Optional inputs
        if (params_.bias)
        {
            info.addInput("bias", params_.bias, 1, params_.n);
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
        info.addScalarInt("device_idx", params_.device_idx);

        return info;
    }

    // =============================================================================
    // FusedQKVGEMMStage Implementation
    // =============================================================================

    FusedQKVGEMMStage::FusedQKVGEMMStage(Params params) : params_(std::move(params)) {}

    bool FusedQKVGEMMStage::execute(IDeviceContext *ctx)
    {
        LOG_DEBUG("[FusedQKVGEMMStage] Execute: m=" << params_.m << " k=" << params_.k
                                                    << " n_q=" << params_.n_q << " n_k=" << params_.n_k << " n_v=" << params_.n_v);

        if (!ctx)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Null device context");
            return false;
        }

        // Validate inputs
        if (!params_.input)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Null input");
            return false;
        }
        if (!params_.wq || !params_.wk || !params_.wv)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Null weight tensor(s)");
            return false;
        }
        if (!params_.output_q || !params_.output_k || !params_.output_v)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Null output buffer(s)");
            return false;
        }
        if (params_.m <= 0 || params_.k <= 0)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Invalid dimensions: m=" << params_.m << " k=" << params_.k);
            return false;
        }

        // Check if outputs are Q8_1 tensors - if so, use Q8_1 execution path
        auto *output_q_q8_1 = dynamic_cast<Q8_1Tensor *>(params_.output_q);
        auto *output_k_q8_1 = dynamic_cast<Q8_1Tensor *>(params_.output_k);
        auto *output_v_q8_1 = dynamic_cast<Q8_1Tensor *>(params_.output_v);

        const bool q8_1_output = (output_q_q8_1 && output_k_q8_1 && output_v_q8_1);

        if (q8_1_output)
        {
            LOG_DEBUG("[FusedQKVGEMMStage] Q8_1 output detected, using Q8_1 execution path");

            // Create FusedGEMM kernel for Q8_1 output support
            FusedGEMM fused_gemm(params_.wq, params_.wk, params_.wv);

            // Check if input is also Q8_1 - use Q8_1→Q8_1 path to avoid double quantization
            auto *input_q8_1 = dynamic_cast<const Q8_1Tensor *>(params_.input);

            if (input_q8_1)
            {
                LOG_DEBUG("[FusedQKVGEMMStage] Q8_1 input detected, using Q8_1→Q8_1 path");

                // Pure Q8_1 path: Q8_1 input → Q8_1 output
                bool success = fused_gemm.execute_q8_1_to_q8_1(
                    input_q8_1->q8_1_blocks(),
                    output_q_q8_1->mutable_q8_1_blocks(),
                    output_k_q8_1->mutable_q8_1_blocks(),
                    output_v_q8_1->mutable_q8_1_blocks(),
                    params_.bias_q, params_.bias_k, params_.bias_v,
                    params_.m, params_.n_q, params_.n_k,
                    params_.k,
                    nullptr, -1); // ctx, device_idx

                if (!success)
                {
                    LOG_ERROR("[FusedQKVGEMMStage] execute_q8_1_to_q8_1 failed");
                    return false;
                }
            }
            else
            {
                LOG_DEBUG("[FusedQKVGEMMStage] FP32 input detected, using FP32→Q8_1 path");

                // FP32 input → Q8_1 output path
                const float *input_fp32 = params_.input->fp32_data();
                if (!input_fp32)
                {
                    LOG_ERROR("[FusedQKVGEMMStage] Failed to get FP32 data from input tensor");
                    return false;
                }

                bool success = fused_gemm.execute_to_q8_1(
                    input_fp32,
                    output_q_q8_1->mutable_q8_1_blocks(),
                    output_k_q8_1->mutable_q8_1_blocks(),
                    output_v_q8_1->mutable_q8_1_blocks(),
                    params_.bias_q, params_.bias_k, params_.bias_v,
                    params_.m, params_.n_q, params_.n_k,
                    params_.k,
                    nullptr, -1); // ctx, device_idx

                if (!success)
                {
                    LOG_ERROR("[FusedQKVGEMMStage] execute_to_q8_1 failed");
                    return false;
                }
            }

            LOG_DEBUG("[FusedQKVGEMMStage] Q8_1 path complete");
            return true;
        }

        // FP32 output path (original implementation)
        // Extract FP32 data from tensors - works for all tensor types via fp32_data()
        const float *input_fp32 = params_.input->fp32_data();
        float *output_q_fp32 = params_.output_q->mutable_data();
        float *output_k_fp32 = params_.output_k->mutable_data();
        float *output_v_fp32 = params_.output_v->mutable_data();

        if (!input_fp32 || !output_q_fp32 || !output_k_fp32 || !output_v_fp32)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Failed to get FP32 data from tensors");
            return false;
        }

        // DEBUG: Log input pointer and first values for parity testing
        LOG_INFO("[FusedQKVGEMMStage] input_fp32 ptr=" << static_cast<const void *>(input_fp32)
                                                       << " input[0:4]=" << std::setprecision(10)
                                                       << input_fp32[0] << "," << input_fp32[1] << ","
                                                       << input_fp32[2] << "," << input_fp32[3]);

        LOG_DEBUG("[FusedQKVGEMMStage] input_type=" << params_.input->dtype_name()
                                                    << " output_type=" << params_.output_q->dtype_name());

        // Get cached kernels from KernelFactory (handles weight packing once)
        auto *gemm_q = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(params_.wq);
        auto *gemm_k = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(params_.wk);
        auto *gemm_v = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(params_.wv);

        if (!gemm_q || !gemm_k || !gemm_v)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Failed to get GEMM kernel(s)");
            return false;
        }

        // Build projection descriptors using device-agnostic interface
        // Pass through bias pointers from params_
        std::vector<ITensorGemm::FusedProjectionDesc> projections = {
            {gemm_q, output_q_fp32, params_.n_q, params_.bias_q, nullptr, false, "Q"},
            {gemm_k, output_k_fp32, params_.n_k, params_.bias_k, nullptr, false, "K"},
            {gemm_v, output_v_fp32, params_.n_v, params_.bias_v, nullptr, false, "V"}};

        // Use the interface method - kernel decides if it can do optimized fusion
        // Note: We use gemm_q's multiply_fused since all kernels should be same type
        bool success = gemm_q->multiply_fused(
            input_fp32,
            projections,
            params_.m,
            params_.k);

        if (!success)
        {
            LOG_ERROR("[FusedQKVGEMMStage] multiply_fused failed");
            return false;
        }

        LOG_DEBUG("[FusedQKVGEMMStage] Complete");
        return true;
    }

    size_t FusedQKVGEMMStage::estimatedFlops() const
    {
        // Three GEMMs: 2 * M * N * K each
        size_t flops_q = static_cast<size_t>(2) * params_.m * params_.n_q * params_.k;
        size_t flops_k = static_cast<size_t>(2) * params_.m * params_.n_k * params_.k;
        size_t flops_v = static_cast<size_t>(2) * params_.m * params_.n_v * params_.k;
        return flops_q + flops_k + flops_v;
    }

    size_t FusedQKVGEMMStage::estimatedMemoryBytes() const
    {
        // Input: m * k reads (shared)
        size_t input_bytes = static_cast<size_t>(params_.m) * params_.k * sizeof(float);

        // Outputs: Q, K, V writes
        size_t output_bytes = static_cast<size_t>(params_.m) * (params_.n_q + params_.n_k + params_.n_v) * sizeof(float);

        // Weight reads (approximate - actual depends on quantization format)
        // Quantized weights are ~4-8 bits/element, but we estimate conservatively
        size_t weight_bytes = static_cast<size_t>(params_.k) * (params_.n_q + params_.n_k + params_.n_v) * sizeof(float) / 4;

        return input_bytes + output_bytes + weight_bytes;
    }

    bool FusedQKVGEMMStage::supportsBackend(ComputeBackendType backend) const
    {
        // Uses ITensorGemm interface - device-agnostic
        // Actual device support depends on the kernel implementation returned by KernelFactory
        switch (backend)
        {
        case ComputeBackendType::CPU:

        case ComputeBackendType::GPU_CUDA:
        case ComputeBackendType::GPU_ROCM:
            return true;
        default:
            return false;
        }
    }

    StageDumpInfo FusedQKVGEMMStage::getDumpInfo() const
    {
        StageDumpInfo info;

        // Input (shared)
        info.addInput("input", params_.input, params_.m, params_.k);

        // Weight tensors
        info.addWeight("wq", params_.wq);
        info.addWeight("wk", params_.wk);
        info.addWeight("wv", params_.wv);

        // Outputs
        info.addOutput("output_q", params_.output_q, params_.m, params_.n_q);
        info.addOutput("output_k", params_.output_k, params_.m, params_.n_k);
        info.addOutput("output_v", params_.output_v, params_.m, params_.n_v);

        // Scalar params
        info.addScalarInt("m", params_.m);
        info.addScalarInt("k", params_.k);
        info.addScalarInt("n_q", params_.n_q);
        info.addScalarInt("n_k", params_.n_k);
        info.addScalarInt("n_v", params_.n_v);

        return info;
    }

    // =============================================================================
    // FusedGateUpGEMMStage Implementation
    // =============================================================================

    FusedGateUpGEMMStage::FusedGateUpGEMMStage(Params params) : params_(std::move(params)) {}

    bool FusedGateUpGEMMStage::execute(IDeviceContext *ctx)
    {
        LOG_DEBUG("[FusedGateUpGEMMStage] Execute: m=" << params_.m << " k=" << params_.k
                                                       << " n_gate=" << params_.n_gate << " n_up=" << params_.n_up);

        if (!ctx)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] Null device context");
            return false;
        }

        // Validate inputs
        if (!params_.input)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] Null input");
            return false;
        }
        if (!params_.w_gate || !params_.w_up)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] Null weight tensor(s)");
            return false;
        }
        if (!params_.output_gate || !params_.output_up)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] Null output buffer(s)");
            return false;
        }
        if (params_.m <= 0 || params_.k <= 0)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] Invalid dimensions: m=" << params_.m << " k=" << params_.k);
            return false;
        }

        LOG_DEBUG("[FusedGateUpGEMMStage] input_type=" << params_.input->dtype_name()
                                                       << " output_gate_type=" << params_.output_gate->dtype_name()
                                                       << " output_up_type=" << params_.output_up->dtype_name());

        // Get cached kernels from KernelFactory (handles weight packing once)
        auto *gemm_gate = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(params_.w_gate);
        auto *gemm_up = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(params_.w_up);

        if (!gemm_gate || !gemm_up)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] Failed to get GEMM kernel(s)");
            return false;
        }

        // Check if outputs are Q8_1 - if so, use multiply_tensor for type-aware dispatch
        // The fused FP32 path doesn't support Q8_1 output directly
        bool gate_is_q8_1 = (params_.output_gate->native_type() == TensorType::Q8_1);
        bool up_is_q8_1 = (params_.output_up->native_type() == TensorType::Q8_1);

        if (gate_is_q8_1 || up_is_q8_1)
        {
            LOG_DEBUG("[FusedGateUpGEMMStage] Using multiply_tensor for Q8_1 output type dispatch");

            // For Q8_1 outputs, we need to use multiply_tensor which handles the type-aware dispatch
            // This means we lose the fusion optimization, but Q8_1 output is more important
            // TODO: Implement fused multiply_tensor_multi for Q8_1 outputs
            // TODO: multiply_tensor doesn't support bias - would need separate bias add pass
            if (params_.bias_gate || params_.bias_up)
            {
                LOG_WARN("[FusedGateUpGEMMStage] Q8_1 output path does not support bias - bias will be ignored!");
            }

            // Gate projection
            bool gate_ok = gemm_gate->multiply_tensor(
                params_.input, params_.output_gate,
                params_.m, params_.n_gate, params_.k,
                false,      // transpose_B
                1.0f, 0.0f, // alpha, beta
                params_.mpi_ctx, params_.device_idx);

            if (!gate_ok)
            {
                LOG_ERROR("[FusedGateUpGEMMStage] Gate GEMM multiply_tensor failed");
                return false;
            }

            // Up projection
            bool up_ok = gemm_up->multiply_tensor(
                params_.input, params_.output_up,
                params_.m, params_.n_up, params_.k,
                false,      // transpose_B
                1.0f, 0.0f, // alpha, beta
                params_.mpi_ctx, params_.device_idx);

            if (!up_ok)
            {
                LOG_ERROR("[FusedGateUpGEMMStage] Up GEMM multiply_tensor failed");
                return false;
            }

            LOG_DEBUG("[FusedGateUpGEMMStage] Q8_1 path complete");
            return true;
        }

        // FP32/BF16/FP16 output path - use optimized fused multiply
        const float *input_fp32 = params_.input->fp32_data();
        float *output_gate_fp32 = params_.output_gate->mutable_data();
        float *output_up_fp32 = params_.output_up->mutable_data();

        if (!input_fp32 || !output_gate_fp32 || !output_up_fp32)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] Failed to get FP32 data from tensors");
            return false;
        }

        // Build projection descriptors using device-agnostic interface
        // Pass through bias pointers from params_
        std::vector<ITensorGemm::FusedProjectionDesc> projections = {
            {gemm_gate, output_gate_fp32, params_.n_gate, params_.bias_gate, nullptr, false, "gate"},
            {gemm_up, output_up_fp32, params_.n_up, params_.bias_up, nullptr, false, "up"}};

        // Use the interface method - kernel decides if it can do optimized fusion
        bool success = gemm_gate->multiply_fused(
            input_fp32,
            projections,
            params_.m,
            params_.k);

        if (!success)
        {
            LOG_ERROR("[FusedGateUpGEMMStage] multiply_fused failed");
            return false;
        }

        LOG_DEBUG("[FusedGateUpGEMMStage] Complete");
        return true;
    }

    size_t FusedGateUpGEMMStage::estimatedFlops() const
    {
        // Two GEMMs: 2 * M * N * K each
        size_t flops_gate = static_cast<size_t>(2) * params_.m * params_.n_gate * params_.k;
        size_t flops_up = static_cast<size_t>(2) * params_.m * params_.n_up * params_.k;
        return flops_gate + flops_up;
    }

    size_t FusedGateUpGEMMStage::estimatedMemoryBytes() const
    {
        // Input: m * k reads (shared)
        size_t input_bytes = static_cast<size_t>(params_.m) * params_.k * sizeof(float);

        // Outputs: gate, up writes
        size_t output_bytes = static_cast<size_t>(params_.m) * (params_.n_gate + params_.n_up) * sizeof(float);

        // Weight reads (approximate - actual depends on quantization format)
        // Quantized weights are ~4-8 bits/element, but we estimate conservatively
        size_t weight_bytes = static_cast<size_t>(params_.k) * (params_.n_gate + params_.n_up) * sizeof(float) / 4;

        return input_bytes + output_bytes + weight_bytes;
    }

    bool FusedGateUpGEMMStage::supportsBackend(ComputeBackendType backend) const
    {
        // Uses ITensorGemm interface - device-agnostic
        // Actual device support depends on the kernel implementation returned by KernelFactory
        switch (backend)
        {
        case ComputeBackendType::CPU:

        case ComputeBackendType::GPU_CUDA:
        case ComputeBackendType::GPU_ROCM:
            return true;
        default:
            return false;
        }
    }

    StageDumpInfo FusedGateUpGEMMStage::getDumpInfo() const
    {
        StageDumpInfo info;

        // Input (shared)
        info.addInput("input", params_.input, params_.m, params_.k);

        // Weight tensors
        info.addWeight("w_gate", params_.w_gate);
        info.addWeight("w_up", params_.w_up);

        // Outputs
        info.addOutput("output_gate", params_.output_gate, params_.m, params_.n_gate);
        info.addOutput("output_up", params_.output_up, params_.m, params_.n_up);

        // Scalar params
        info.addScalarInt("m", params_.m);
        info.addScalarInt("k", params_.k);
        info.addScalarInt("n_gate", params_.n_gate);
        info.addScalarInt("n_up", params_.n_up);

        return info;
    }

    // =============================================================================
    // RMSNormStage Implementation (Type-Safe via IActivationTensor)
    // =============================================================================

    RMSNormStage::RMSNormStage(Params params) : params_(std::move(params)) {}

    bool RMSNormStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[RMSNormStage] Null device context");
            return false;
        }

        if (!params_.input || !params_.output || !params_.gamma)
        {
            LOG_ERROR("[RMSNormStage] Null tensor(s): input=" << params_.input
                                                              << " output=" << params_.output
                                                              << " gamma=" << params_.gamma);
            return false;
        }

        // Use explicit seq_len if provided, otherwise derive from tensor dimensions
        // CRITICAL: For pre-allocated buffers, params_.seq_len must be set to avoid
        // processing garbage data beyond the actual sequence
        const int seq_len = params_.seq_len > 0 ? params_.seq_len : static_cast<int>(params_.input->rows());
        const int hidden_dim = static_cast<int>(params_.input->cols());

        LOG_DEBUG("[RMSNormStage] Execute: seq_len=" << seq_len
                                                     << " hidden_dim=" << hidden_dim
                                                     << " eps=" << params_.eps
                                                     << " tensor_type=" << params_.input->dtype_name());

        // Create kernel via KernelFactory with automatic type dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(params_.device_idx);
        auto kernel = llaminar::v2::kernels::KernelFactory::createRMSNorm(params_.input, dev_type);
        if (!kernel)
        {
            LOG_ERROR("[RMSNormStage] Failed to create RMSNorm kernel for type "
                      << params_.input->dtype_name());
            return false;
        }

        // apply_tensor handles input != output cases internally
        return kernel->apply_tensor(
            params_.input,
            params_.gamma,
            params_.output,
            seq_len,
            hidden_dim,
            params_.eps,
            params_.mpi_ctx,
            params_.device_idx);
    }

    size_t RMSNormStage::estimatedFlops() const
    {
        if (!params_.input)
            return 0;

        const int seq_len = params_.seq_len > 0 ? params_.seq_len : static_cast<int>(params_.input->rows());
        const int hidden_dim = static_cast<int>(params_.input->cols());
        // Per row: hidden_dim squares + hidden_dim adds + sqrt + div + hidden_dim muls
        // Approximately 4 * hidden_dim FLOPs per row
        return static_cast<size_t>(4) * seq_len * hidden_dim;
    }

    size_t RMSNormStage::estimatedMemoryBytes() const
    {
        if (!params_.input)
            return 0;

        const int seq_len = params_.seq_len > 0 ? params_.seq_len : static_cast<int>(params_.input->rows());
        const int hidden_dim = static_cast<int>(params_.input->cols());
        // Read input + gamma, write output (in-place, so same buffer)
        size_t input_bytes = static_cast<size_t>(seq_len) * hidden_dim * sizeof(float);
        size_t gamma_bytes = static_cast<size_t>(hidden_dim) * sizeof(float);
        return 2 * input_bytes + gamma_bytes; // Read + write + gamma
    }

    bool RMSNormStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:

            return true;
        default:
            return false;
        }
    }

    StageDumpInfo RMSNormStage::getDumpInfo() const
    {
        StageDumpInfo info;

        if (!params_.input)
            return info;

        const int seq_len = params_.seq_len > 0 ? params_.seq_len : static_cast<int>(params_.input->rows());
        const int hidden_dim = static_cast<int>(params_.input->cols());

        // Use TensorBase* overload for type-safe FP32 extraction (handles Q8_1)
        info.addInput("input", params_.input, seq_len, hidden_dim);

        // Gamma weights
        if (params_.gamma)
        {
            info.addInput("gamma", params_.gamma, 1, hidden_dim);
        }

        // Output - use TensorBase* overload
        info.addOutput("output", params_.output, seq_len, hidden_dim);

        // Scalar params
        info.addScalarInt("seq_len", seq_len);
        info.addScalarInt("hidden_dim", hidden_dim);
        info.addScalar("eps", params_.eps);

        return info;
    }

    // =============================================================================
    // RoPEStage Implementation (Type-Safe via KernelFactory)
    // =============================================================================

    RoPEStage::RoPEStage(Params params) : params_(std::move(params)) {}

    bool RoPEStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[RoPEStage] Null device context");
            return false;
        }

        if (!params_.Q)
        {
            LOG_ERROR("[RoPEStage] Null Q tensor");
            return false;
        }

        // Get seq_len from tensor dimensions
        // Q tensor is [seq_len, n_heads * head_dim]
        const int seq_len = static_cast<int>(params_.Q->rows());

        LOG_DEBUG("[RoPEStage] Execute: seq_len=" << seq_len
                                                  << " n_heads=" << params_.n_heads
                                                  << " n_kv_heads=" << params_.n_kv_heads
                                                  << " head_dim=" << params_.head_dim
                                                  << " pos_offset=" << params_.pos_offset
                                                  << " tensor_type=" << params_.Q->dtype_name());

        // Create kernel via KernelFactory with automatic type dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(params_.device_idx);
        auto kernel = llaminar::v2::kernels::KernelFactory::createRoPE(params_.Q, dev_type);
        if (!kernel)
        {
            LOG_ERROR("[RoPEStage] Failed to create RoPE kernel for type "
                      << params_.Q->dtype_name());
            return false;
        }

        // Generate position_ids array [pos_offset, pos_offset+1, ..., pos_offset+seq_len-1]
        std::vector<int> position_ids(seq_len);
        for (int i = 0; i < seq_len; ++i)
        {
            position_ids[i] = params_.pos_offset + i;
        }

        const int n_kv_heads = params_.n_kv_heads > 0 ? params_.n_kv_heads : params_.n_heads;

        // Apply RoPE via kernel's apply_tensor method
        return kernel->apply_tensor(
            params_.Q,
            params_.K, // May be nullptr
            position_ids.data(),
            seq_len,
            params_.n_heads,
            n_kv_heads,
            params_.head_dim,
            params_.theta_base,
            params_.mpi_ctx,
            params_.device_idx);
    }

    size_t RoPEStage::estimatedFlops() const
    {
        if (!params_.Q)
            return 0;

        const int seq_len = static_cast<int>(params_.Q->rows());
        // Per position per head: head_dim/2 rotations, each ~10 FLOPs (sin, cos, 4 muls, 2 adds)
        size_t flops = static_cast<size_t>(10) * seq_len * params_.n_heads * (params_.head_dim / 2);
        if (params_.K)
        {
            int n_kv_heads = params_.n_kv_heads > 0 ? params_.n_kv_heads : params_.n_heads;
            flops += static_cast<size_t>(10) * seq_len * n_kv_heads * (params_.head_dim / 2);
        }
        return flops;
    }

    size_t RoPEStage::estimatedMemoryBytes() const
    {
        if (!params_.Q)
            return 0;

        const int seq_len = static_cast<int>(params_.Q->rows());
        size_t bytes = static_cast<size_t>(2) * seq_len * params_.n_heads *
                       params_.head_dim * sizeof(float); // Q read + write
        if (params_.K)
        {
            int n_kv_heads = params_.n_kv_heads > 0 ? params_.n_kv_heads : params_.n_heads;
            bytes += static_cast<size_t>(2) * seq_len * n_kv_heads * params_.head_dim * sizeof(float);
        }
        return bytes;
    }

    bool RoPEStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:

            return true;
        default:
            return false;
        }
    }

    StageDumpInfo RoPEStage::getDumpInfo() const
    {
        StageDumpInfo info;
        if (!params_.Q)
            return info;

        const int seq_len = static_cast<int>(params_.Q->rows());

        // Q tensor (in-place operation) - use safe accessor for Q8_1 compatibility
        const float *q_data = getSafeFp32Data(params_.Q);
        if (q_data)
        {
            info.addInput("Q", q_data,
                          seq_len, params_.n_heads * params_.head_dim);
            info.addOutput("Q", q_data,
                           seq_len, params_.n_heads * params_.head_dim);
        }

        // K tensor (optional, in-place)
        if (params_.K)
        {
            int n_kv_heads = params_.n_kv_heads > 0 ? params_.n_kv_heads : params_.n_heads;
            const float *k_data = getSafeFp32Data(params_.K);
            if (k_data)
            {
                info.addInput("K", k_data,
                              seq_len, n_kv_heads * params_.head_dim);
                info.addOutput("K", k_data,
                               seq_len, n_kv_heads * params_.head_dim);
            }
        }

        // Scalar params
        info.addScalarInt("seq_len", seq_len);
        info.addScalarInt("n_heads", params_.n_heads);
        info.addScalarInt("n_kv_heads", params_.n_kv_heads);
        info.addScalarInt("head_dim", params_.head_dim);
        info.addScalarInt("pos_offset", params_.pos_offset);
        info.addScalar("theta_base", params_.theta_base);

        return info;
    }

    // =============================================================================
    // AttentionStage Implementation
    // =============================================================================

    AttentionStage::AttentionStage(Params params) : params_(std::move(params)) {}

    bool AttentionStage::execute(IDeviceContext *ctx)
    {
        LOG_DEBUG("[AttentionStage] Execute: seq_len=" << params_.seq_len << " kv_len=" << params_.kv_len
                                                       << " n_heads=" << params_.n_heads << " n_kv_heads=" << params_.n_kv_heads
                                                       << " head_dim=" << params_.head_dim << " causal=" << params_.causal);

        if (!ctx)
        {
            LOG_ERROR("[AttentionStage] Null device context");
            return false;
        }

        // This is a simplified implementation - production would use optimized kernels
        const float *Q = static_cast<const float *>(params_.Q);
        const float *K = static_cast<const float *>(params_.K);
        const float *V = static_cast<const float *>(params_.V);
        float *output = static_cast<float *>(params_.output);

        // Debug: dump first few values of Q/K/V
        LOG_DEBUG("[AttentionStage] Q[0:4] = " << Q[0] << ", " << Q[1] << ", " << Q[2] << ", " << Q[3]);
        LOG_DEBUG("[AttentionStage] K[0:4] = " << K[0] << ", " << K[1] << ", " << K[2] << ", " << K[3]);
        LOG_DEBUG("[AttentionStage] V[0:4] = " << V[0] << ", " << V[1] << ", " << V[2] << ", " << V[3]);

        const int seq_len = params_.seq_len;
        const int kv_len = params_.kv_len;
        const int n_heads = params_.n_heads;
        const int n_kv_heads = params_.n_kv_heads;
        const int head_dim = params_.head_dim;
        const int heads_per_kv = n_heads / n_kv_heads;
        const float scale = params_.scale;

        // Get workspace from context for attention scores
        size_t scores_size = static_cast<size_t>(seq_len) * kv_len * sizeof(float);
        void *workspace = ctx->getWorkspace(scores_size * n_heads);
        float *scores_buf = static_cast<float *>(workspace);

        // Process each query head
        ctx->runFor(0, static_cast<size_t>(n_heads), [=](size_t h_)
                    {
        int h = static_cast<int>(h_);
        int kv_h = h / heads_per_kv;  // GQA: map query head to KV head
        float* scores = scores_buf + h * seq_len * kv_len;
        
        // Q * K^T
        for (int q_pos = 0; q_pos < seq_len; ++q_pos) {
            const float* q_vec = Q + q_pos * n_heads * head_dim + h * head_dim;
            
            for (int k_pos = 0; k_pos < kv_len; ++k_pos) {
                // Apply causal mask
                if (params_.causal && k_pos > q_pos) {
                    scores[q_pos * kv_len + k_pos] = -INFINITY;
                    continue;
                }
                
                const float* k_vec = K + k_pos * n_kv_heads * head_dim + kv_h * head_dim;
                
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += q_vec[d] * k_vec[d];
                }
                scores[q_pos * kv_len + k_pos] = dot * scale;
            }
        }
        
        // Softmax
        for (int q_pos = 0; q_pos < seq_len; ++q_pos) {
            float* row = scores + q_pos * kv_len;
            
            // Find max
            float max_val = row[0];
            for (int k_pos = 1; k_pos < kv_len; ++k_pos) {
                if (row[k_pos] > max_val) max_val = row[k_pos];
            }
            
            // Exp and sum
            float sum = 0.0f;
            for (int k_pos = 0; k_pos < kv_len; ++k_pos) {
                row[k_pos] = std::exp(row[k_pos] - max_val);
                sum += row[k_pos];
            }
            
            // Normalize
            float inv_sum = 1.0f / sum;
            for (int k_pos = 0; k_pos < kv_len; ++k_pos) {
                row[k_pos] *= inv_sum;
            }
        }
        
        // Scores * V
        for (int q_pos = 0; q_pos < seq_len; ++q_pos) {
            float* out_vec = output + q_pos * n_heads * head_dim + h * head_dim;
            const float* score_row = scores + q_pos * kv_len;
            
            std::memset(out_vec, 0, head_dim * sizeof(float));
            
            for (int k_pos = 0; k_pos < kv_len; ++k_pos) {
                const float* v_vec = V + k_pos * n_kv_heads * head_dim + kv_h * head_dim;
                float s = score_row[k_pos];
                
                for (int d = 0; d < head_dim; ++d) {
                    out_vec[d] += s * v_vec[d];
                }
            }
        } });

        // Debug: dump first few values of output
        LOG_DEBUG("[AttentionStage] output[0:4] = " << output[0] << ", " << output[1] << ", " << output[2] << ", " << output[3]);

        return true;
    }

    size_t AttentionStage::estimatedFlops() const
    {
        // QK: 2 * seq_len * kv_len * head_dim (per head)
        // Softmax: ~5 * seq_len * kv_len (per head)
        // V: 2 * seq_len * kv_len * head_dim (per head)
        size_t qk_flops = 2ULL * params_.seq_len * params_.kv_len * params_.head_dim;
        size_t softmax_flops = 5ULL * params_.seq_len * params_.kv_len;
        size_t v_flops = 2ULL * params_.seq_len * params_.kv_len * params_.head_dim;
        return (qk_flops + softmax_flops + v_flops) * params_.n_heads;
    }

    size_t AttentionStage::estimatedMemoryBytes() const
    {
        size_t q_bytes = static_cast<size_t>(params_.seq_len) * params_.n_heads *
                         params_.head_dim * sizeof(float);
        size_t kv_bytes = static_cast<size_t>(params_.kv_len) * params_.n_kv_heads *
                          params_.head_dim * sizeof(float);
        size_t out_bytes = q_bytes;
        return q_bytes + 2 * kv_bytes + out_bytes; // Q + K + V + output
    }

    bool AttentionStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:

            return true;
        default:
            return false;
        }
    }

    StageDumpInfo AttentionStage::getDumpInfo() const
    {
        StageDumpInfo info;

        // Input tensors
        info.addInput("Q", static_cast<const float *>(params_.Q),
                      params_.seq_len, params_.n_heads * params_.head_dim);
        info.addInput("K", static_cast<const float *>(params_.K),
                      params_.kv_len, params_.n_kv_heads * params_.head_dim);
        info.addInput("V", static_cast<const float *>(params_.V),
                      params_.kv_len, params_.n_kv_heads * params_.head_dim);

        // Output
        info.addOutput("output", static_cast<const float *>(params_.output),
                       params_.seq_len, params_.n_heads * params_.head_dim);

        // Scalar params
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("kv_len", params_.kv_len);
        info.addScalarInt("n_heads", params_.n_heads);
        info.addScalarInt("n_kv_heads", params_.n_kv_heads);
        info.addScalarInt("head_dim", params_.head_dim);
        info.addScalarBool("causal", params_.causal);
        info.addScalar("scale", params_.scale);

        return info;
    }

    // =============================================================================
    // SwiGLUStage Implementation (Type-Safe via TensorBase)
    // =============================================================================

    SwiGLUStage::SwiGLUStage(Params params) : params_(std::move(params)) {}

    bool SwiGLUStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[SwiGLUStage] Null device context");
            return false;
        }

        if (!params_.gate || !params_.up || !params_.output)
        {
            LOG_ERROR("[SwiGLUStage] Null tensor(s): gate=" << params_.gate
                                                            << " up=" << params_.up
                                                            << " output=" << params_.output);
            return false;
        }

        // Get dimensions from tensor
        const int seq_len = static_cast<int>(params_.gate->rows());
        const int intermediate_dim = static_cast<int>(params_.gate->cols());

        LOG_DEBUG("[SwiGLUStage] Execute: seq_len=" << seq_len
                                                    << " intermediate_dim=" << intermediate_dim
                                                    << " tensor_type=" << params_.gate->dtype_name());

        // Create kernel via KernelFactory with automatic type dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(params_.device_idx);
        auto kernel = llaminar::v2::kernels::KernelFactory::createSwiGLU(params_.gate, dev_type);
        if (!kernel)
        {
            LOG_ERROR("[SwiGLUStage] Failed to create SwiGLU kernel for type "
                      << params_.gate->dtype_name());
            return false;
        }

        // Apply SwiGLU via kernel's apply_tensor method
        return kernel->apply_tensor(
            params_.gate,
            params_.up,
            params_.output,
            seq_len,
            intermediate_dim,
            false, // add_residual
            params_.mpi_ctx,
            params_.device_idx);
    }

    size_t SwiGLUStage::estimatedFlops() const
    {
        if (!params_.gate)
            return 0;

        const int seq_len = static_cast<int>(params_.gate->rows());
        const int intermediate_dim = static_cast<int>(params_.gate->cols());
        // Per element: exp, div, mul, mul (~6 FLOPs)
        return static_cast<size_t>(6) * seq_len * intermediate_dim;
    }

    size_t SwiGLUStage::estimatedMemoryBytes() const
    {
        if (!params_.gate)
            return 0;

        const int seq_len = static_cast<int>(params_.gate->rows());
        const int intermediate_dim = static_cast<int>(params_.gate->cols());
        size_t bytes = static_cast<size_t>(seq_len) * intermediate_dim * sizeof(float);
        return 3 * bytes; // gate + up + output
    }

    bool SwiGLUStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:

            return true;
        default:
            return false;
        }
    }

    StageDumpInfo SwiGLUStage::getDumpInfo() const
    {
        StageDumpInfo info;

        if (!params_.gate || !params_.up || !params_.output)
            return info;

        const int seq_len = static_cast<int>(params_.gate->rows());
        const int intermediate_dim = static_cast<int>(params_.gate->cols());

        // Use TensorBase* overload for type-safe FP32 extraction (handles Q8_1)
        info.addInput("gate", params_.gate, seq_len, intermediate_dim);
        info.addInput("up", params_.up, seq_len, intermediate_dim);

        // Output - use TensorBase* overload
        info.addOutput("output", params_.output, seq_len, intermediate_dim);

        // Scalar params
        info.addScalarInt("seq_len", seq_len);
        info.addScalarInt("intermediate_dim", intermediate_dim);

        return info;
    }

    // =============================================================================
    // ResidualAddStage Implementation (Type-Safe via TensorBase)
    // =============================================================================

    ResidualAddStage::ResidualAddStage(Params params) : params_(std::move(params)) {}

    bool ResidualAddStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[ResidualAddStage] Null device context");
            return false;
        }

        if (!params_.input || !params_.residual || !params_.output)
        {
            LOG_ERROR("[ResidualAddStage] Null tensor(s): input=" << params_.input
                                                                  << " residual=" << params_.residual
                                                                  << " output=" << params_.output);
            return false;
        }

        // Use explicit num_elements if provided, otherwise fall back to input->numel()
        // CRITICAL: For decode mode with pre-allocated buffers, params_.num_elements must be set
        // to avoid reading/writing beyond the actual sequence data.
        const size_t num_elements = params_.num_elements > 0 ? params_.num_elements : params_.input->numel();

        LOG_DEBUG("[ResidualAddStage] Execute: num_elements=" << num_elements
                                                              << " tensor_type=" << params_.input->dtype_name());

        // Dispatch based on tensor type, passing num_elements through
        TensorType ttype = params_.input->native_type();
        switch (ttype)
        {
        case TensorType::FP32:
            return executeFP32(ctx, num_elements);
        case TensorType::BF16:
            return executeBF16(ctx, num_elements);
        case TensorType::FP16:
            return executeFP16(ctx, num_elements);
        case TensorType::Q8_1:
            return executeQ8_1(ctx, num_elements);
        default:
            LOG_ERROR("[ResidualAddStage] Unsupported tensor type: " << params_.input->dtype_name());
            return false;
        }
    }

    bool ResidualAddStage::executeFP32(IDeviceContext *ctx, size_t num_elements)
    {
        const float *input = params_.input->data();
        const float *residual = params_.residual->data();
        float *output = params_.output->mutable_data();
        const size_t n = num_elements;

        LOG_DEBUG("[ResidualAddStage::FP32] input[0:4]="
                  << input[0] << "," << input[1] << "," << input[2] << "," << input[3]
                  << " residual[0:4]="
                  << residual[0] << "," << residual[1] << "," << residual[2] << "," << residual[3]);

        ctx->runFor(0, n, [=](size_t i)
                    { output[i] = input[i] + residual[i]; });

        LOG_DEBUG("[ResidualAddStage::FP32] output[0:4]="
                  << output[0] << "," << output[1] << "," << output[2] << "," << output[3]);

        return true;
    }

    bool ResidualAddStage::executeBF16(IDeviceContext *ctx, size_t num_elements)
    {
        const uint16_t *input = static_cast<const uint16_t *>(params_.input->raw_data());
        const uint16_t *residual = static_cast<const uint16_t *>(params_.residual->raw_data());
        uint16_t *output = static_cast<uint16_t *>(params_.output->raw_mutable_data());
        const size_t n = num_elements;

        LOG_DEBUG("[ResidualAddStage::BF16] Converting and adding " << n << " elements");

        ctx->runFor(0, n, [=](size_t i)
                    {
            float in_f = simd::bf16_to_fp32(input[i]);
            float res_f = simd::bf16_to_fp32(residual[i]);
            output[i] = simd::fp32_to_bf16(in_f + res_f); });

        return true;
    }

    bool ResidualAddStage::executeFP16(IDeviceContext *ctx, size_t num_elements)
    {
        const uint16_t *input = static_cast<const uint16_t *>(params_.input->raw_data());
        const uint16_t *residual = static_cast<const uint16_t *>(params_.residual->raw_data());
        uint16_t *output = static_cast<uint16_t *>(params_.output->raw_mutable_data());
        const size_t n = num_elements;

        LOG_DEBUG("[ResidualAddStage::FP16] Converting and adding " << n << " elements");

        ctx->runFor(0, n, [=](size_t i)
                    {
            float in_f = simd::fp16_to_fp32(input[i]);
            float res_f = simd::fp16_to_fp32(residual[i]);
            output[i] = simd::fp32_to_fp16(in_f + res_f); });

        return true;
    }

    bool ResidualAddStage::executeQ8_1(IDeviceContext *ctx, size_t num_elements)
    {
        // Cast to Q8_1Tensor to access block storage
        const auto *input_q8 = dynamic_cast<const Q8_1Tensor *>(params_.input);
        const auto *residual_q8 = dynamic_cast<const Q8_1Tensor *>(params_.residual);
        auto *output_q8 = dynamic_cast<Q8_1Tensor *>(params_.output);

        if (!input_q8 || !residual_q8 || !output_q8)
        {
            LOG_ERROR("[ResidualAddStage::Q8_1] Failed to cast tensors to Q8_1Tensor");
            return false;
        }

        const size_t numel = num_elements;
        if (numel % 32 != 0)
        {
            LOG_ERROR("[ResidualAddStage::Q8_1] Element count " << numel << " is not a multiple of 32");
            return false;
        }

        const Q8_1Block *input_blocks = input_q8->q8_1_blocks();
        const Q8_1Block *residual_blocks = residual_q8->q8_1_blocks();
        Q8_1Block *output_blocks = output_q8->mutable_q8_1_blocks();

        LOG_DEBUG("[ResidualAddStage::Q8_1] Adding " << numel << " elements ("
                                                     << (numel / 32) << " blocks)");

        // Use the SIMD-optimized Q8_1 addition (AVX512/AVX2/scalar)
        simd::q8_1_add_q8_1(input_blocks, residual_blocks, output_blocks, numel);

        return true;
    }

    size_t ResidualAddStage::estimatedFlops() const
    {
        if (!params_.input)
            return 0;

        // For BF16/FP16: convert (1) + add + convert (1) = ~3 ops/elem
        // For FP32: 1 add per element
        // For Q8_1: dequant (2) + add + requant (2) = ~5 ops/elem
        TensorType ttype = params_.input->native_type();
        if (ttype == TensorType::BF16 || ttype == TensorType::FP16)
        {
            return params_.input->numel() * 3;
        }
        if (ttype == TensorType::Q8_1)
        {
            return params_.input->numel() * 5;
        }
        return params_.input->numel();
    }

    size_t ResidualAddStage::estimatedMemoryBytes() const
    {
        if (!params_.input)
            return 0;

        // Memory depends on tensor type
        TensorType ttype = params_.input->native_type();

        // Q8_1: 1 byte per element + 4 bytes per 32-element block for scale+sum
        // ~1.125 bytes/element on average
        if (ttype == TensorType::Q8_1)
        {
            // Q8_1Block: 32 bytes qs + 2 bytes d + 2 bytes sum_qs = 36 bytes per 32 elements
            size_t num_blocks = (params_.input->numel() + 31) / 32;
            return 3 * num_blocks * sizeof(Q8_1Block); // input + residual + output
        }

        size_t bytes_per_element;
        switch (ttype)
        {
        case TensorType::BF16:
        case TensorType::FP16:
            bytes_per_element = 2;
            break;
        case TensorType::FP32:
        default:
            bytes_per_element = 4;
            break;
        }
        return 3 * params_.input->numel() * bytes_per_element; // input + residual + output
    }

    bool ResidualAddStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:

            return true;
        default:
            return false;
        }
    }

    StageDumpInfo ResidualAddStage::getDumpInfo() const
    {
        StageDumpInfo info;

        if (!params_.input || !params_.residual || !params_.output)
            return info;

        // Determine dtype string based on tensor type
        const char *dtype = params_.input->dtype_name();
        size_t elem_size = sizeof(float);
        TensorType ttype = params_.input->native_type();
        switch (ttype)
        {
        case TensorType::BF16:
        case TensorType::FP16:
            elem_size = 2;
            break;
        case TensorType::Q8_1:
            // Q8_1Block: 36 bytes per 32 elements = 1.125 bytes/elem
            // For dump purposes, use 1 (closest integer approximation)
            elem_size = 1;
            break;
        default:
            break;
        }

        int rows = static_cast<int>(params_.input->rows());
        int cols = static_cast<int>(params_.input->cols());

        // Input tensors - use raw_data() for void* compatibility
        info.inputs.push_back({"input", params_.input->raw_data(),
                               static_cast<size_t>(rows), static_cast<size_t>(cols),
                               dtype, elem_size});
        info.inputs.push_back({"residual", params_.residual->raw_data(),
                               static_cast<size_t>(rows), static_cast<size_t>(cols),
                               dtype, elem_size});

        // Output
        info.outputs.push_back({"output", params_.output->raw_data(),
                                static_cast<size_t>(rows), static_cast<size_t>(cols),
                                dtype, elem_size});

        // Scalar params
        info.addScalarInt("num_elements", static_cast<int>(params_.input->numel()));
        info.addScalarInt("rows", rows);
        info.addScalarInt("cols", cols);

        return info;
    }

    // =============================================================================
    // AllreduceStage Implementation
    // =============================================================================

    AllreduceStage::AllreduceStage(Params params) : params_(std::move(params)) {}

    bool AllreduceStage::execute(IDeviceContext *ctx)
    {
        (void)ctx;
        LOG_DEBUG("[AllreduceStage] Execute: buffer=" << params_.buffer
                                                      << " count=" << params_.count << " has_comm=" << (params_.mpi_comm != nullptr));
        if (!params_.mpi_comm)
        {
            LOG_ERROR("[AllreduceStage] Null MPI communicator");
            return false;
        }

        MPI_Comm comm = static_cast<MPI_Comm>(params_.mpi_comm);

        LOG_DEBUG("[AllreduceStage] Calling MPI_Allreduce with count=" << params_.count);
        int result = MPI_Allreduce(
            MPI_IN_PLACE,
            params_.buffer,
            static_cast<int>(params_.count),
            MPI_FLOAT,
            MPI_SUM,
            comm);

        LOG_DEBUG("[AllreduceStage] MPI_Allreduce returned " << result);
        return result == MPI_SUCCESS;
    }

    bool AllreduceStage::supportsBackend(ComputeBackendType backend) const
    {
        // Allreduce is backend-agnostic (works with any device that has MPI support)
        (void)backend;
        return true;
    }

    // =============================================================================
    // MoE Stages Implementation
    // =============================================================================

    MoERouterStage::MoERouterStage(Params params) : params_(std::move(params)) {}

    bool MoERouterStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[MoERouterStage] Null device context");
            return false;
        }

        // Router is a simple matmul: hidden @ gate_weights
        // This computes logits for each expert
        const float *hidden = static_cast<const float *>(params_.hidden);
        const float *gate_weights = static_cast<const float *>(params_.gate_weights);
        float *logits = params_.router_logits;

        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int num_experts = params_.num_experts;

        ctx->runFor(0, static_cast<size_t>(seq_len), [=](size_t t_)
                    {
        int t = static_cast<int>(t_);
        const float* h = hidden + t * d_model;
        float* out = logits + t * num_experts;
        
        for (int e = 0; e < num_experts; ++e) {
            const float* w = gate_weights + e * d_model;
            float dot = 0.0f;
            for (int d = 0; d < d_model; ++d) {
                dot += h[d] * w[d];
            }
            out[e] = dot;
        } });
        return true;
    }

    size_t MoERouterStage::estimatedFlops() const
    {
        // seq_len * d_model * num_experts (dot products)
        return static_cast<size_t>(2) * params_.seq_len * params_.d_model * params_.num_experts;
    }

    bool MoERouterStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:

            return true;
        default:
            return false;
        }
    }

    // -----------------------------------------------------------------------------

    MoEExpertStage::MoEExpertStage(Params params) : params_(std::move(params)) {}

    bool MoEExpertStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[MoEExpertStage] Null device context");
            return false;
        }

        if (!params_.token_indices || params_.token_indices->empty())
        {
            // No tokens routed to this expert - nothing to do
            return true;
        }

        // This is a placeholder - real implementation would use the actual expert weights
        // For now, we just demonstrate the structure
        LOG_DEBUG("[MoEExpertStage] Processing expert " << params_.expert_id
                                                        << " with " << params_.token_indices->size() << " tokens");

        // In real implementation:
        // 1. Gather tokens from input based on token_indices
        // 2. Apply gate projection
        // 3. Apply up projection
        // 4. SwiGLU activation
        // 5. Apply down projection
        // 6. Scatter results back

        return true;
    }

    std::string MoEExpertStage::name() const
    {
        std::ostringstream oss;
        oss << "MOE_EXPERT_" << params_.expert_id;
        return oss.str();
    }

    size_t MoEExpertStage::estimatedFlops() const
    {
        if (!params_.token_indices)
            return 0;
        size_t num_tokens = params_.token_indices->size();
        // FFN: gate + up + down projections
        // gate: num_tokens * d_model * intermediate_dim
        // up: num_tokens * d_model * intermediate_dim
        // down: num_tokens * intermediate_dim * d_model
        return static_cast<size_t>(6) * num_tokens * params_.d_model * params_.intermediate_dim;
    }

    bool MoEExpertStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:

            return true;
        default:
            return false;
        }
    }

    // -----------------------------------------------------------------------------

    MoECombineStage::MoECombineStage(Params params) : params_(std::move(params)) {}

    bool MoECombineStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[MoECombineStage] Null device context");
            return false;
        }

        // Placeholder - combines expert outputs weighted by router scores
        LOG_DEBUG("[MoECombineStage] Combining "
                  << (params_.expert_outputs ? params_.expert_outputs->size() : 0)
                  << " expert outputs");

        // In real implementation:
        // For each token:
        //   output[t] = sum over k experts: weight[t][k] * expert_output[k][t]

        return true;
    }

    bool MoECombineStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:

            return true;
        default:
            return false;
        }
    }

    // =============================================================================
    // AttentionWithKVCacheStage Implementation
    // =============================================================================

    AttentionWithKVCacheStage::AttentionWithKVCacheStage(Params params)
        : params_(std::move(params)) {}

    AttentionWithKVCacheStage::Mode AttentionWithKVCacheStage::effectiveMode() const
    {
        if (params_.mode != Mode::AUTO)
        {
            return params_.mode;
        }

        // Auto-detect based on seq_len and cache state
        if (params_.batch_size > 1 && params_.sequence_lengths != nullptr)
        {
            return Mode::BATCHED;
        }

        // Single sequence: check if decode (single token) vs prefill
        if (params_.seq_len == 1)
        {
            // Single token with cache -> decode mode
            if (params_.kv_cache != nullptr)
            {
                int cached = params_.kv_cache->get_cached_tokens(params_.layer_idx, 0);
                if (cached > 0)
                {
                    return Mode::DECODE;
                }
            }
        }

        return Mode::PREFILL;
    }

    MpiAttentionConfig AttentionWithKVCacheStage::buildAttentionConfig() const
    {
        MpiAttentionConfig config;
        config.n_heads = params_.n_heads;
        config.n_kv_heads = params_.n_kv_heads;
        config.head_dim = params_.head_dim;
        config.causal = params_.causal;
        config.window_size = params_.window_size;

        // Auto-detect precision from Q tensor type (Q is authoritative since K/V come from cache)
        if (params_.Q && params_.Q->native_type() == TensorType::Q8_1)
        {
            config.precision = ActivationPrecision::Q8_1;
        }
        else
        {
            config.precision = ActivationPrecision::FP32;
        }

        config.mpi_ctx = params_.mpi_ctx;
        config.mpi_strategy = static_cast<MPIStrategy>(params_.mpi_strategy);
        config.verbose_logging = false;

        // Wire workspace buffers if provided
        if (params_.workspace_scores)
        {
            config.workspace_scores = std::shared_ptr<TensorBase>(
                params_.workspace_scores, [](TensorBase *) {}); // Non-owning
        }
        if (params_.workspace_context)
        {
            config.workspace_context = std::shared_ptr<TensorBase>(
                params_.workspace_context, [](TensorBase *) {});
        }
        if (params_.workspace_mask)
        {
            config.workspace_mask = std::shared_ptr<TensorBase>(
                params_.workspace_mask, [](TensorBase *) {});
        }

        return config;
    }

    bool AttentionWithKVCacheStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[AttentionWithKVCacheStage] Null device context");
            return false;
        }

        if (!params_.Q || !params_.K || !params_.V || !params_.output)
        {
            LOG_ERROR("[AttentionWithKVCacheStage] Invalid tensors: Q=" << params_.Q
                                                                        << " K=" << params_.K << " V=" << params_.V << " output=" << params_.output);
            return false;
        }

        Mode mode = effectiveMode();
        LOG_DEBUG("[AttentionWithKVCacheStage] Execute mode=" << static_cast<int>(mode)
                                                              << " seq_len=" << params_.seq_len << " batch=" << params_.batch_size
                                                              << " layer=" << params_.layer_idx << " heads=" << params_.n_heads
                                                              << " kv_heads=" << params_.n_kv_heads << " head_dim=" << params_.head_dim);

        switch (mode)
        {
        case Mode::PREFILL:
            return executePrefill(ctx);
        case Mode::DECODE:
            return executeDecode(ctx);
        case Mode::BATCHED:
            return executeBatched(ctx);
        case Mode::AUTO:
            // Should never reach here
            LOG_ERROR("[AttentionWithKVCacheStage] AUTO mode not resolved");
            return false;
        }

        return false;
    }

    bool AttentionWithKVCacheStage::executePrefill(IDeviceContext *ctx)
    {
        (void)ctx; // Device context for future GPU support

        LOG_DEBUG("[AttentionWithKVCacheStage::executePrefill] layer=" << params_.layer_idx
                                                                       << " seq_len=" << params_.seq_len);

        // Step 1: Append new K/V to cache (if cache provided)
        if (params_.kv_cache)
        {
            bool append_ok = params_.kv_cache->append_kv(
                params_.layer_idx, 0, params_.K, params_.V, params_.seq_len);
            if (!append_ok)
            {
                LOG_ERROR("[AttentionWithKVCacheStage] Failed to append K/V to cache");
                return false;
            }
            LOG_DEBUG("[AttentionWithKVCacheStage] Appended " << params_.seq_len
                                                              << " tokens to layer " << params_.layer_idx << " cache");
        }

        // Step 2: Get full K/V from cache (includes just-appended tokens)
        TensorBase *K_full = params_.K;
        TensorBase *V_full = params_.V;
        int kv_len = params_.seq_len;

        if (params_.kv_cache)
        {
            // Use cached K/V (will include all tokens including just-appended)
            K_full = params_.kv_cache->get_k_base(params_.layer_idx, 0);
            V_full = params_.kv_cache->get_v_base(params_.layer_idx, 0);
            kv_len = params_.kv_cache->get_cached_tokens(params_.layer_idx, 0);
            LOG_DEBUG("[AttentionWithKVCacheStage] Using cached K/V with " << kv_len << " tokens");
        }

        // Step 3: Create tensor views with actual seq_len dimensions
        // This is critical because activation buffers are allocated at max_seq_len
        // but MpiAttentionOrchestrator validates tensor shapes match
        int q_dim = params_.n_heads * params_.head_dim;
        int kv_dim = params_.n_kv_heads * params_.head_dim;

        auto Q_view = params_.Q->create_view({static_cast<size_t>(params_.seq_len), static_cast<size_t>(q_dim)});
        auto K_view = K_full->create_view({static_cast<size_t>(kv_len), static_cast<size_t>(kv_dim)});
        auto V_view = V_full->create_view({static_cast<size_t>(kv_len), static_cast<size_t>(kv_dim)});
        auto out_view = params_.output->create_view({static_cast<size_t>(params_.seq_len), static_cast<size_t>(q_dim)});

        if (!Q_view || !K_view || !V_view || !out_view)
        {
            LOG_ERROR("[AttentionWithKVCacheStage] Failed to create tensor views");
            return false;
        }

        // Step 4: Build attention config and dispatch to MpiAttentionOrchestrator
        MpiAttentionConfig config = buildAttentionConfig();

        // Dispatch to MPI-aware attention using views
        bool success = MpiAttentionOrchestrator::compute_mpi(
            Q_view.get(), K_view.get(), V_view.get(), out_view.get(),
            config,
            params_.batch_size,
            params_.sequence_lengths);

        if (!success)
        {
            LOG_ERROR("[AttentionWithKVCacheStage] MpiAttentionOrchestrator::compute_mpi failed");
            return false;
        }

        LOG_DEBUG("[AttentionWithKVCacheStage::executePrefill] Complete");
        return true;
    }

    bool AttentionWithKVCacheStage::executeDecode(IDeviceContext *ctx)
    {
        (void)ctx;

        LOG_DEBUG("[AttentionWithKVCacheStage::executeDecode] layer=" << params_.layer_idx
                                                                      << " position=" << params_.position_offset);

        if (!params_.kv_cache)
        {
            LOG_ERROR("[AttentionWithKVCacheStage::executeDecode] Decode mode requires KV cache");
            return false;
        }

        // Step 1: Append single token K/V to cache
        bool append_ok = params_.kv_cache->append_kv(
            params_.layer_idx, 0, params_.K, params_.V, 1);
        if (!append_ok)
        {
            LOG_ERROR("[AttentionWithKVCacheStage] Failed to append decode token to cache");
            return false;
        }

        // Step 2: Get full cached K/V
        TensorBase *K_cached = params_.kv_cache->get_k_base(params_.layer_idx, 0);
        TensorBase *V_cached = params_.kv_cache->get_v_base(params_.layer_idx, 0);
        int kv_len = params_.kv_cache->get_cached_tokens(params_.layer_idx, 0);

        LOG_DEBUG("[AttentionWithKVCacheStage::executeDecode] Attending to " << kv_len << " cached tokens");

        // Step 3: Create tensor views with actual dimensions
        // Q is single token (decode), K/V is full cache length
        int q_dim = params_.n_heads * params_.head_dim;
        int kv_dim = params_.n_kv_heads * params_.head_dim;
        const int q_seq_len = 1; // Decode mode: single query token

        auto Q_view = params_.Q->create_view({static_cast<size_t>(q_seq_len), static_cast<size_t>(q_dim)});
        auto K_view = K_cached->create_view({static_cast<size_t>(kv_len), static_cast<size_t>(kv_dim)});
        auto V_view = V_cached->create_view({static_cast<size_t>(kv_len), static_cast<size_t>(kv_dim)});
        auto out_view = params_.output->create_view({static_cast<size_t>(q_seq_len), static_cast<size_t>(q_dim)});

        if (!Q_view || !K_view || !V_view || !out_view)
        {
            LOG_ERROR("[AttentionWithKVCacheStage::executeDecode] Failed to create tensor views");
            return false;
        }

        // Step 4: Create attention kernel via KernelFactory (device-aware dispatch)
        // This uses the typed kernel path (CPUAttentionKernelTyped) which properly
        // handles decode mode where Q.seq_len != K.seq_len
        const int device_idx = params_.device_idx;
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_idx);
        auto attention_kernel = llaminar::v2::kernels::KernelFactory::createAttention(Q_view.get(), dev_type);

        if (!attention_kernel)
        {
            LOG_ERROR("[AttentionWithKVCacheStage::executeDecode] Failed to create attention kernel");
            return false;
        }

        // Step 5: Allocate workspace for attention scores [n_heads, q_seq_len, kv_len]
        // Use simple allocation - in production, use pre-allocated workspace
        FP32Tensor scores_workspace({static_cast<size_t>(params_.n_heads * q_seq_len * kv_len)});

        // Step 6: Build causal mask for decode
        // For decode: Q is at position kv_len-1 (end of sequence), attends to [0, kv_len-1]
        FP32Tensor mask_tensor({static_cast<size_t>(q_seq_len * kv_len)});
        float *mask_data = mask_tensor.mutable_data();

        for (int q = 0; q < q_seq_len; ++q)
        {
            int q_pos = (kv_len - q_seq_len) + q; // Q position in full sequence
            for (int k = 0; k < kv_len; ++k)
            {
                // Causal: Q at position q_pos can attend to K at positions [0, q_pos]
                mask_data[q * kv_len + k] = (k <= q_pos) ? 0.0f : -std::numeric_limits<float>::infinity();
            }
        }

        // Step 7: Dispatch via compute_tensor which handles decode mode (q_seq_len != kv_len)
        bool success = attention_kernel->compute_tensor(
            Q_view.get(), K_view.get(), V_view.get(), out_view.get(),
            params_.batch_size,
            q_seq_len, // Query sequence length (1 for decode)
            kv_len,    // Key/value sequence length (full cache)
            params_.n_heads,
            params_.n_kv_heads,
            params_.head_dim,
            true, // causal
            -1,   // window_size (disabled)
            &scores_workspace,
            &mask_tensor,
            params_.mpi_ctx.get(),
            device_idx);

        if (!success)
        {
            LOG_ERROR("[AttentionWithKVCacheStage::executeDecode] Attention kernel compute_tensor failed");
            return false;
        }

        LOG_DEBUG("[AttentionWithKVCacheStage::executeDecode] Complete");
        return true;
    }

    bool AttentionWithKVCacheStage::executeBatched(IDeviceContext *ctx)
    {
        (void)ctx;

        LOG_DEBUG("[AttentionWithKVCacheStage::executeBatched] layer=" << params_.layer_idx
                                                                       << " batch_size=" << params_.batch_size << " seq_len=" << params_.seq_len);

        // For batched mode with different sequence lengths, we need to:
        // 1. Append K/V per sequence to cache
        // 2. Build combined K/V tensors with padding
        // 3. Apply padding-aware attention mask

        if (params_.kv_cache)
        {
            // Append each sequence's K/V to cache
            const int d_kv = params_.n_kv_heads * params_.head_dim;
            for (int b = 0; b < params_.batch_size; ++b)
            {
                int actual_len = params_.sequence_lengths ? (*params_.sequence_lengths)[b] : params_.seq_len;

                // Create view of K/V for this batch
                // Note: This assumes K/V are [batch * seq_len, n_kv_heads * head_dim]
                // TODO: Implement proper batch slicing

                bool append_ok = params_.kv_cache->append_kv(
                    params_.layer_idx, b, params_.K, params_.V, actual_len);
                if (!append_ok)
                {
                    LOG_ERROR("[AttentionWithKVCacheStage::executeBatched] Failed to append batch " << b);
                    return false;
                }
            }
        }

        // Build attention config
        MpiAttentionConfig config = buildAttentionConfig();

        // Dispatch batched attention
        bool success = MpiAttentionOrchestrator::compute_mpi(
            params_.Q, params_.K, params_.V, params_.output,
            config,
            params_.batch_size,
            params_.sequence_lengths);

        if (!success)
        {
            LOG_ERROR("[AttentionWithKVCacheStage::executeBatched] MpiAttentionOrchestrator failed");
            return false;
        }

        LOG_DEBUG("[AttentionWithKVCacheStage::executeBatched] Complete");
        return true;
    }

    size_t AttentionWithKVCacheStage::estimatedFlops() const
    {
        // QK: 2 * seq_len * kv_len * head_dim (per head)
        // For prefill: kv_len ≈ seq_len
        // For decode: kv_len = cached_tokens, seq_len = 1
        int estimated_kv_len = params_.seq_len; // Conservative estimate
        if (params_.kv_cache && params_.seq_len == 1)
        {
            estimated_kv_len = params_.kv_cache->get_cached_tokens(params_.layer_idx, 0);
        }

        size_t qk_flops = 2ULL * params_.seq_len * estimated_kv_len * params_.head_dim;
        size_t softmax_flops = 5ULL * params_.seq_len * estimated_kv_len;
        size_t v_flops = 2ULL * params_.seq_len * estimated_kv_len * params_.head_dim;
        return (qk_flops + softmax_flops + v_flops) * params_.n_heads * params_.batch_size;
    }

    size_t AttentionWithKVCacheStage::estimatedMemoryBytes() const
    {
        // Q + K + V + output (all at FP32)
        size_t qkv_bytes = static_cast<size_t>(params_.batch_size) * params_.seq_len *
                           (params_.n_heads + 2 * params_.n_kv_heads) * params_.head_dim * sizeof(float);
        return qkv_bytes;
    }

    bool AttentionWithKVCacheStage::supportsBackend(ComputeBackendType backend) const
    {
        // MpiAttentionOrchestrator currently only supports CPU
        switch (backend)
        {
        case ComputeBackendType::CPU:

            return true;
        default:
            return false;
        }
    }

    StageDumpInfo AttentionWithKVCacheStage::getDumpInfo() const
    {
        StageDumpInfo info;

        // Q/K/V inputs - use safe accessor for Q8_1 compatibility
        if (params_.Q)
        {
            const float *q_data = getSafeFp32Data(params_.Q);
            if (q_data)
            {
                info.addInput("Q", q_data,
                              params_.batch_size * params_.seq_len,
                              params_.n_heads * params_.head_dim);
            }
        }

        // Output - use safe accessor for Q8_1 compatibility
        if (params_.output)
        {
            const float *out_data = getSafeFp32Data(params_.output);
            if (out_data)
            {
                info.addOutput("output", out_data,
                               params_.batch_size * params_.seq_len,
                               params_.n_heads * params_.head_dim);
            }
        }

        // Scalar params
        info.addScalarInt("batch_size", params_.batch_size);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("n_heads", params_.n_heads);
        info.addScalarInt("n_kv_heads", params_.n_kv_heads);
        info.addScalarInt("head_dim", params_.head_dim);
        info.addScalarBool("causal", params_.causal);

        return info;
    }

    // =============================================================================
    // KVCacheAppendStage Implementation
    // =============================================================================

    KVCacheAppendStage::KVCacheAppendStage(Params params)
        : params_(std::move(params)) {}

    bool KVCacheAppendStage::execute(IDeviceContext *ctx)
    {
        (void)ctx;

        if (!params_.kv_cache)
        {
            LOG_ERROR("[KVCacheAppendStage] No KV cache provided");
            return false;
        }

        if (!params_.K || !params_.V)
        {
            LOG_ERROR("[KVCacheAppendStage] Invalid K/V tensors");
            return false;
        }

        int num_tokens = params_.num_tokens;
        if (num_tokens <= 0)
        {
            // Infer from tensor shape
            num_tokens = static_cast<int>(params_.K->shape()[0]);
        }

        LOG_DEBUG("[KVCacheAppendStage] Appending " << num_tokens
                                                    << " tokens to layer " << params_.layer_idx << " seq " << params_.seq_idx);

        bool success = params_.kv_cache->append_kv(
            params_.layer_idx, params_.seq_idx,
            params_.K, params_.V, num_tokens);

        if (!success)
        {
            LOG_ERROR("[KVCacheAppendStage] append_kv failed");
            return false;
        }

        return true;
    }

    // =============================================================================
    // AttentionComputeStage Implementation
    // =============================================================================

    AttentionComputeStage::AttentionComputeStage(Params params)
        : params_(std::move(params)) {}

    bool AttentionComputeStage::execute(IDeviceContext *ctx)
    {
        // Detect attention mode if auto-detection enabled
        AttentionMode mode = params_.attention_mode;
        if (params_.auto_detect_mode)
        {
            mode = detect_attention_mode(params_.batch_size, params_.seq_len, params_.kv_len);
        }

        LOG_DEBUG("[AttentionComputeStage] Execute: batch=" << params_.batch_size
                                                            << " seq_len=" << params_.seq_len
                                                            << " kv_len=" << params_.kv_len
                                                            << " n_heads=" << params_.n_heads
                                                            << " n_kv_heads=" << params_.n_kv_heads
                                                            << " head_dim=" << params_.head_dim
                                                            << " mode=" << attention_mode_name(mode));

        // Validate inputs
        if (!params_.Q || !params_.K || !params_.V || !params_.output)
        {
            LOG_ERROR("[AttentionComputeStage] Null tensor pointers");
            return false;
        }

        if (params_.seq_len <= 0 || params_.kv_len <= 0 ||
            params_.n_heads <= 0 || params_.n_kv_heads <= 0 || params_.head_dim <= 0)
        {
            LOG_ERROR("[AttentionComputeStage] Invalid dimensions");
            return false;
        }

        if (params_.n_heads % params_.n_kv_heads != 0)
        {
            LOG_ERROR("[AttentionComputeStage] n_heads (" << params_.n_heads
                                                          << ") must be divisible by n_kv_heads (" << params_.n_kv_heads << ")");
            return false;
        }

        // Determine device type from context
        using DeviceType = llaminar::v2::kernels::DeviceType;
        DeviceType dev_type = DeviceType::CPU;

        if (ctx)
        {
            auto *gpu_ctx = dynamic_cast<IGPUDeviceContext *>(ctx);
            if (gpu_ctx)
            {
#if defined(HAVE_CUDA)
                dev_type = DeviceType::CUDA;
#elif defined(HAVE_ROCM)
                dev_type = DeviceType::ROCm;
#endif
            }
        }

        // Create attention kernel via KernelFactory
        auto kernel = llaminar::v2::kernels::KernelFactory::createAttention(params_.Q, dev_type);
        if (!kernel)
        {
            LOG_ERROR("[AttentionComputeStage] Failed to create attention kernel for tensor type "
                      << params_.Q->dtype_name());
            return false;
        }

        // Get device index
        int device_idx = params_.device_idx;
        if (device_idx < 0 && ctx)
        {
            device_idx = ctx->deviceIndex();
        }

        // Dispatch to kernel's compute_tensor() method with detected mode
        // The kernel uses mode internally for optimized dispatch (decode vs prefill path)
        bool success = kernel->compute_tensor(
            params_.Q, params_.K, params_.V, params_.output,
            params_.batch_size,
            params_.seq_len,
            params_.kv_len,
            params_.n_heads,
            params_.n_kv_heads,
            params_.head_dim,
            params_.causal,
            params_.window_size,
            params_.workspace_scores,
            params_.workspace_mask,
            params_.mpi_ctx,
            device_idx);

        if (!success)
        {
            LOG_ERROR("[AttentionComputeStage] Kernel compute_tensor() failed");
            return false;
        }

        LOG_DEBUG("[AttentionComputeStage] Execute complete (mode=" << attention_mode_name(mode) << ")");
        return true;
    }

    size_t AttentionComputeStage::estimatedFlops() const
    {
        // Attention FLOPs:
        // Q @ K^T: 2 * batch * n_heads * seq_len * kv_len * head_dim
        // softmax: ~4 * batch * n_heads * seq_len * kv_len
        // scores @ V: 2 * batch * n_heads * seq_len * kv_len * head_dim
        const size_t qk_flops = 2ULL * params_.batch_size * params_.n_heads *
                                params_.seq_len * params_.kv_len * params_.head_dim;
        const size_t softmax_flops = 4ULL * params_.batch_size * params_.n_heads *
                                     params_.seq_len * params_.kv_len;
        const size_t sv_flops = qk_flops;
        return qk_flops + softmax_flops + sv_flops;
    }

    size_t AttentionComputeStage::estimatedMemoryBytes() const
    {
        // Workspace for attention scores: n_heads * seq_len * kv_len
        return static_cast<size_t>(params_.n_heads) * params_.seq_len * params_.kv_len * sizeof(float);
    }

    bool AttentionComputeStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:

            return true;
#if defined(HAVE_CUDA)
        case ComputeBackendType::GPU_CUDA:
            return true;
#endif
#if defined(HAVE_ROCM)
        case ComputeBackendType::GPU_ROCM:
            return true;
#endif
        default:
            return false;
        }
    }

    StageDumpInfo AttentionComputeStage::getDumpInfo() const
    {
        StageDumpInfo info;

        // Note: TensorBase* doesn't expose data() directly - we rely on execute()
        // to handle the actual type dispatch. For dump purposes, just report dimensions.
        // TODO: Add addWeight() variant that accepts TensorBase* for type-safe dump

        // Scalars capture all necessary info for debugging
        info.addScalarInt("batch_size", params_.batch_size);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("kv_len", params_.kv_len);
        info.addScalarInt("n_heads", params_.n_heads);
        info.addScalarInt("n_kv_heads", params_.n_kv_heads);
        info.addScalarInt("head_dim", params_.head_dim);
        info.addScalarBool("causal", params_.causal);
        info.addScalarInt("window_size", params_.window_size);
        info.addScalarInt("device_idx", params_.device_idx);

        // Add attention mode info (as int - PREFILL=0, DECODE=1, BATCHED_DECODE=2, CHUNKED_PREFILL=3)
        AttentionMode mode = params_.auto_detect_mode
                                 ? detect_attention_mode(params_.batch_size, params_.seq_len, params_.kv_len)
                                 : params_.attention_mode;
        info.addScalarInt("attention_mode", static_cast<int>(mode));
        info.addScalarBool("auto_detect_mode", params_.auto_detect_mode);

        return info;
    }

    // =============================================================================
    // EmbeddingStage Implementation
    // =============================================================================

    EmbeddingStage::EmbeddingStage(Params params) : params_(std::move(params)) {}

    bool EmbeddingStage::execute(IDeviceContext *ctx)
    {
        LOG_DEBUG("[EmbeddingStage] Execute: num_tokens=" << params_.num_tokens
                                                          << " d_model=" << params_.d_model
                                                          << " vocab_size=" << params_.vocab_size);

        // Validate inputs
        if (!params_.embed_table || !params_.output)
        {
            LOG_ERROR("[EmbeddingStage] Null tensor pointers");
            return false;
        }

        if (!params_.token_ids && !params_.token_batches)
        {
            LOG_ERROR("[EmbeddingStage] No token input provided");
            return false;
        }

        if (params_.num_tokens <= 0 || params_.d_model <= 0)
        {
            LOG_ERROR("[EmbeddingStage] Invalid dimensions");
            return false;
        }

        // Use the same approach as EmbeddingOp in Qwen2Pipeline:
        // Get the embedding kernel from the output tensor (IActivationTensor::createEmbedding)
        // This handles automatic type dispatch for FP32, Q8_1, etc.
        auto *activation = dynamic_cast<IActivationTensor *>(params_.output);
        if (!activation)
        {
            LOG_ERROR("[EmbeddingStage] Output tensor must implement IActivationTensor");
            return false;
        }

        auto kernel = activation->createEmbedding();
        if (!kernel)
        {
            LOG_ERROR("[EmbeddingStage] Failed to create embedding kernel");
            return false;
        }

        // Handle batched vs single sequence input
        if (params_.token_batches)
        {
            // Batched input: flatten token batches with padding
            const int batch_size = static_cast<int>(params_.token_batches->size());
            const int padded_len = params_.padded_seq_len;
            const int total_positions = batch_size * padded_len;

            std::vector<int> flat_token_ids(total_positions, 0); // Zero-initialized for padding

            int global_idx = 0;
            for (int b = 0; b < batch_size; ++b)
            {
                const auto &tokens = (*params_.token_batches)[b];
                const int seq_len = static_cast<int>(tokens.size());

                // Copy actual tokens
                for (int i = 0; i < seq_len && i < padded_len; ++i)
                {
                    flat_token_ids[global_idx + i] = tokens[i];
                }
                global_idx += padded_len;
            }

            // Delegate to kernel's apply_tensor - handles all type dispatch internally
            if (!kernel->apply_tensor(params_.embed_table, flat_token_ids.data(), total_positions,
                                      params_.d_model, params_.output, params_.mpi_ctx, params_.device_idx))
            {
                LOG_ERROR("[EmbeddingStage] Kernel apply_tensor failed (batched)");
                return false;
            }

            // Zero out padding positions in output
            // This matches what EmbeddingOp does in its execute_batched
            global_idx = 0;
            for (int b = 0; b < batch_size; ++b)
            {
                const auto &tokens = (*params_.token_batches)[b];
                const int seq_len = static_cast<int>(tokens.size());

                for (int i = seq_len; i < padded_len; ++i)
                {
                    zero_output_row(params_.output, global_idx + i, params_.d_model);
                }
                global_idx += padded_len;
            }
        }
        else
        {
            // Single sequence input: delegate directly to kernel
            if (!kernel->apply_tensor(params_.embed_table, params_.token_ids, params_.num_tokens,
                                      params_.d_model, params_.output, params_.mpi_ctx, params_.device_idx))
            {
                LOG_ERROR("[EmbeddingStage] Kernel apply_tensor failed");
                return false;
            }
        }

        return true;
    }

    /**
     * @brief Zero out a row of the output tensor (for padding)
     *
     * Handles both FP32 and Q8_1 output formats.
     */
    void EmbeddingStage::zero_output_row(TensorBase *output, int row_idx, int d_model)
    {
        TensorType output_type = output->native_type();

        if (output_type == TensorType::FP32)
        {
            float *data = const_cast<float *>(output->fp32_data());
            if (data)
            {
                std::memset(data + row_idx * d_model, 0, d_model * sizeof(float));
            }
        }
        else if (output_type == TensorType::Q8_1)
        {
            // For Q8_1, zero out the blocks for this row
            auto *q8_1_output = dynamic_cast<Q8_1Tensor *>(output);
            if (q8_1_output)
            {
                // Q8_1 has 32 elements per block
                const int block_size = 32;
                const int blocks_per_row = (d_model + block_size - 1) / block_size;
                Q8_1Block *blocks = q8_1_output->mutable_q8_1_blocks();

                if (blocks)
                {
                    for (int b = 0; b < blocks_per_row; ++b)
                    {
                        Q8_1Block &block = blocks[row_idx * blocks_per_row + b];
                        block.d = 0.0f;
                        std::memset(block.qs, 0, 32);
                    }
                }
            }
        }
    }

    size_t EmbeddingStage::estimatedFlops() const
    {
        // Embedding is pure memory ops, no FLOPs
        return 0;
    }

    size_t EmbeddingStage::estimatedMemoryBytes() const
    {
        // Read: embed_table[token_id] for each token
        // Write: output[num_tokens, d_model]
        return params_.num_tokens * params_.d_model * sizeof(float) * 2;
    }

    bool EmbeddingStage::supportsBackend(ComputeBackendType backend) const
    {
        // Embedding is currently CPU-only (memcpy-based)
        return backend == ComputeBackendType::CPU;
    }

    StageDumpInfo EmbeddingStage::getDumpInfo() const
    {
        StageDumpInfo info;

        if (params_.output)
        {
            info.addOutput("embeddings", params_.output,
                           params_.num_tokens, params_.d_model);
        }

        info.addScalarInt("num_tokens", params_.num_tokens);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarInt("vocab_size", params_.vocab_size);
        info.addScalarInt("device_idx", params_.device_idx);

        return info;
    }

    // =============================================================================
    // LMHeadStage Implementation
    // =============================================================================

    LMHeadStage::LMHeadStage(Params params) : params_(std::move(params)) {}

    bool LMHeadStage::execute(IDeviceContext *ctx)
    {
        LOG_DEBUG("[LMHeadStage] Execute: seq_len=" << params_.seq_len
                                                    << " d_model=" << params_.d_model
                                                    << " vocab_size=" << params_.vocab_size);

        // Validate inputs
        if (!params_.hidden_states || !params_.lm_head_weight || !params_.logits)
        {
            LOG_ERROR("[LMHeadStage] Null tensor pointers");
            return false;
        }

        if (params_.seq_len <= 0 || params_.d_model <= 0 || params_.vocab_size <= 0)
        {
            LOG_ERROR("[LMHeadStage] Invalid dimensions");
            return false;
        }

        // Get or create GEMM kernel for LM head weight
        // The kernel handles quantized weights and typed activations
        ITensorGemm *lm_gemm = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(params_.lm_head_weight);
        if (!lm_gemm)
        {
            LOG_ERROR("[LMHeadStage] Failed to get/create LM head GEMM kernel");
            return false;
        }

        // LM head: logits = hidden @ lm_head^T
        // hidden: [seq_len, d_model], lm_head: [vocab_size, d_model]
        // output: [seq_len, vocab_size]
        bool success = lm_gemm->multiply_tensor(
            params_.hidden_states,
            params_.logits,
            params_.seq_len,
            params_.vocab_size,
            params_.d_model,
            true, // transpose_B (lm_head is [vocab_size, d_model])
            1.0f, 0.0f,
            params_.mpi_ctx,
            params_.device_idx);

        if (!success)
        {
            LOG_ERROR("[LMHeadStage] GEMM failed");
            return false;
        }

        // Apply bias if present
        if (params_.bias)
        {
            float *logits_data = const_cast<float *>(params_.logits->fp32_data());
            if (!logits_data)
            {
                LOG_ERROR("[LMHeadStage] Failed to get logits data for bias add");
                return false;
            }

// Add bias to each row
#pragma omp parallel for
            for (int s = 0; s < params_.seq_len; ++s)
            {
                float *row = logits_data + s * params_.vocab_size;
                for (int v = 0; v < params_.vocab_size; ++v)
                {
                    row[v] += params_.bias[v];
                }
            }
        }

        return true;
    }

    size_t LMHeadStage::estimatedFlops() const
    {
        // GEMM: 2 * seq_len * vocab_size * d_model
        return 2ULL * params_.seq_len * params_.vocab_size * params_.d_model;
    }

    size_t LMHeadStage::estimatedMemoryBytes() const
    {
        // Read: hidden [seq_len, d_model], lm_head [vocab_size, d_model]
        // Write: logits [seq_len, vocab_size]
        size_t hidden_bytes = params_.seq_len * params_.d_model * sizeof(float);
        size_t weight_bytes = params_.vocab_size * params_.d_model * sizeof(float); // Approximation for quantized
        size_t output_bytes = params_.seq_len * params_.vocab_size * sizeof(float);
        return hidden_bytes + weight_bytes + output_bytes;
    }

    bool LMHeadStage::supportsBackend(ComputeBackendType backend) const
    {
        // LM head uses KernelFactory which supports CPU and GPU
        switch (backend)
        {
        case ComputeBackendType::CPU:
        case ComputeBackendType::GPU_CUDA:
        case ComputeBackendType::GPU_ROCM:
            return true;
        default:
            return false;
        }
    }

    StageDumpInfo LMHeadStage::getDumpInfo() const
    {
        StageDumpInfo info;

        if (params_.hidden_states)
        {
            info.addInput("hidden_states", params_.hidden_states,
                          params_.seq_len, params_.d_model);
        }

        if (params_.lm_head_weight)
        {
            info.addWeight("lm_head_weight", params_.lm_head_weight);
        }

        if (params_.logits)
        {
            info.addOutput("logits", params_.logits,
                           params_.seq_len, params_.vocab_size);
        }

        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarInt("vocab_size", params_.vocab_size);
        info.addScalarBool("has_bias", params_.bias != nullptr);
        info.addScalarInt("device_idx", params_.device_idx);

        return info;
    }

    // =============================================================================
    // ComputeStageFactory Implementation
    // =============================================================================

    std::unique_ptr<IComputeStage> ComputeStageFactory::createGEMM(
        const GEMMStage::Params &params)
    {
        // Unified: GEMMStage handles all backends via KernelFactory dispatch at execute-time
        return std::make_unique<GEMMStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createFusedQKVGEMM(
        const FusedQKVGEMMStage::Params &params)
    {
        // Unified: FusedQKVGEMMStage will use KernelFactory at execute-time
        return std::make_unique<FusedQKVGEMMStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createFusedGateUpGEMM(
        const FusedGateUpGEMMStage::Params &params)
    {
        // Unified: FusedGateUpGEMMStage will use KernelFactory at execute-time
        return std::make_unique<FusedGateUpGEMMStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createRMSNorm(
        const RMSNormStage::Params &params)
    {
        // Unified: RMSNormStage uses KernelFactory at execute-time for device dispatch
        return std::make_unique<RMSNormStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createRoPE(
        const RoPEStage::Params &params)
    {
        // Unified: RoPEStage uses KernelFactory at execute-time for device dispatch
        return std::make_unique<RoPEStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createAttention(
        const AttentionStage::Params &params)
    {
        // Unified: AttentionStage uses KernelFactory at execute-time for device dispatch
        return std::make_unique<AttentionStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createSwiGLU(
        const SwiGLUStage::Params &params)
    {
        // Unified: SwiGLUStage uses KernelFactory at execute-time for device dispatch
        return std::make_unique<SwiGLUStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createResidualAdd(
        const ResidualAddStage::Params &params)
    {
        // Unified: ResidualAddStage uses KernelFactory at execute-time for device dispatch
        return std::make_unique<ResidualAddStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoERouter(
        const MoERouterStage::Params &params)
    {
        // Unified: MoERouterStage will use KernelFactory at execute-time
        return std::make_unique<MoERouterStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoEExpert(
        const MoEExpertStage::Params &params)
    {
        // Unified: MoEExpertStage will use KernelFactory at execute-time
        return std::make_unique<MoEExpertStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoECombine(
        const MoECombineStage::Params &params)
    {
        // Unified: MoECombineStage will use KernelFactory at execute-time
        return std::make_unique<MoECombineStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createAllreduce(
        const AllreduceStage::Params &params)
    {
        // Allreduce is backend-agnostic (uses MPI directly)
        return std::make_unique<AllreduceStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createAttentionWithKVCache(
        const AttentionWithKVCacheStage::Params &params)
    {
        // Unified: AttentionWithKVCacheStage uses KernelFactory at execute-time
        return std::make_unique<AttentionWithKVCacheStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createKVCacheAppend(
        const KVCacheAppendStage::Params &params)
    {
        // KV cache append is backend-agnostic (pure memory operations)
        return std::make_unique<KVCacheAppendStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createAttentionCompute(
        const AttentionComputeStage::Params &params)
    {
        // Unified: AttentionComputeStage uses KernelFactory at execute-time
        return std::make_unique<AttentionComputeStage>(params);
    }

    // =============================================================================
    // Model-Level Stage Factories
    // =============================================================================

    std::unique_ptr<IComputeStage> ComputeStageFactory::createEmbedding(
        const EmbeddingStage::Params &params)
    {
        return std::make_unique<EmbeddingStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createLMHead(
        const LMHeadStage::Params &params)
    {
        return std::make_unique<LMHeadStage>(params);
    }

} // namespace llaminar2
