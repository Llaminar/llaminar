/**
 * @file ComputeStage.cpp
 * @brief Implementation of compute stage abstractions
 * @author David Sanftenberg
 * @date December 2025
 */

#include "ComputeStage.h"
#include "BufferRole.h"
#include "../utils/Logger.h"
#include "../utils/DebugEnv.h"
#include "../utils/OpenMPUtils.h"
#include "../kernels/KernelFactory.h"
#include "../kernels/cpu/gemm_v4/QuantisedGemmKernel.h"
#include "../kernels/cpu/gemm_v4/FusedGEMM.h"
#include "../kernels/cpu/attention/q8_1/FusedAttentionWoKernel.h" // For FusedAttentionWoStage
#include "../tensors/UnifiedKVCache.h"
#include "../tensors/SIMDHelpers.h"
#include "../tensors/Tensors.h" // For TensorBase::rows(), cols(), dtype_name()
#include "../tensors/TensorSlice.h"
#include "../utils/MPIStrategy.h"
#include "../utils/MPIContext.h"

#include <cstring>
#include <cmath>
#include <sstream>
#include <iomanip>
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
        case ComputeStageType::FUSED_ATTENTION_WO:
            return "FUSED_ATTENTION_WO";
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
    // IComputeStage Tracing Implementation (Task 3: Stage Tracing Infrastructure)
    // Controlled by LLAMINAR_TRACE_STAGES and related env vars
    // =============================================================================

    bool IComputeStage::shouldTrace() const
    {
        const auto &cfg = debugEnv().execution;
        if (!cfg.trace_stages)
        {
            return false;
        }

        // Check filter if specified
        if (!cfg.trace_filter.empty())
        {
            // Simple substring match on stage name
            return name().find(cfg.trace_filter) != std::string::npos;
        }

        return true;
    }

    std::string IComputeStage::formatFloatArray(const float *data, size_t count)
    {
        const auto &cfg = debugEnv().execution;
        const int sample_count = std::min(static_cast<size_t>(cfg.trace_sample_count), count);

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6) << "[";

        for (int i = 0; i < sample_count; ++i)
        {
            if (i > 0)
                oss << ", ";
            oss << data[i];
        }

        if (count > static_cast<size_t>(sample_count))
        {
            oss << ", ... (" << count << " total)";
        }
        oss << "]";

        return oss.str();
    }

    float IComputeStage::computeChecksum(const float *data, size_t count)
    {
        // Simple Kahan summation for numerical stability
        float sum = 0.0f;
        float c = 0.0f; // Compensation for lost low-order bits

        for (size_t i = 0; i < count; ++i)
        {
            float y = data[i] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }

        return sum;
    }

    void IComputeStage::traceInput(const std::string &tensor_name, const TensorBase *tensor) const
    {
        if (!shouldTrace() || !tensor)
            return;

        const auto &cfg = debugEnv().execution;

        // Safe data access - numel() and fp32_data() may return nullptr for released tensors
        size_t count = 0;
        try
        {
            count = tensor->numel();
            if (count == 0)
            {
                LOG_INFO("[TRACE] " << name() << " INPUT '" << tensor_name
                                    << "' [empty tensor]");
                return;
            }
        }
        catch (...)
        {
            LOG_WARN("[TRACE] " << name() << " INPUT '" << tensor_name
                                << "' [numel() failed]");
            return;
        }

        const float *data = nullptr;
        try
        {
            data = tensor->fp32_data();
        }
        catch (const std::exception &e)
        {
            LOG_WARN("[TRACE] " << name() << " INPUT '" << tensor_name
                                << "' [fp32_data() failed: " << e.what() << "]");
            return;
        }
        catch (...)
        {
            LOG_WARN("[TRACE] " << name() << " INPUT '" << tensor_name
                                << "' [fp32_data() failed: unknown error]");
            return;
        }

        // Handle nullptr - tensor's raw data may have been released (e.g., after GEMM packing)
        if (!data)
        {
            std::ostringstream oss;
            oss << "[TRACE] " << name() << " INPUT '" << tensor_name << "' ";
            if (cfg.trace_shapes)
            {
                oss << "[" << tensor->rows() << "x" << tensor->cols() << " "
                    << tensor->dtype_name() << "] ";
            }
            oss << "[data unavailable - raw weights released after GEMM packing]";
            LOG_INFO(oss.str());
            return;
        }

        std::ostringstream oss;
        oss << "[TRACE] " << name() << " INPUT '" << tensor_name << "' ";

        if (cfg.trace_shapes)
        {
            oss << "[" << tensor->rows() << "x" << tensor->cols() << " "
                << tensor->dtype_name() << "] ";
        }

        oss << formatFloatArray(data, count);

        if (cfg.trace_checksums)
        {
            oss << " checksum=" << computeChecksum(data, count);
        }

        LOG_INFO(oss.str());
    }

    void IComputeStage::traceOutput(const std::string &tensor_name, const TensorBase *tensor) const
    {
        if (!shouldTrace() || !tensor)
            return;

        const auto &cfg = debugEnv().execution;

        // Safe data access - numel() and fp32_data() may crash for invalid/empty tensors
        size_t count = 0;
        try
        {
            count = tensor->numel();
            if (count == 0)
            {
                LOG_INFO("[TRACE] " << name() << " OUTPUT '" << tensor_name
                                    << "' [empty tensor]");
                return;
            }
        }
        catch (...)
        {
            LOG_WARN("[TRACE] " << name() << " OUTPUT '" << tensor_name
                                << "' [numel() failed]");
            return;
        }

        const float *data = nullptr;
        try
        {
            data = tensor->fp32_data();
        }
        catch (const std::exception &e)
        {
            LOG_WARN("[TRACE] " << name() << " OUTPUT '" << tensor_name
                                << "' [fp32_data() failed: " << e.what() << "]");
            return;
        }
        catch (...)
        {
            LOG_WARN("[TRACE] " << name() << " OUTPUT '" << tensor_name
                                << "' [fp32_data() failed: unknown error]");
            return;
        }

        if (!data)
        {
            LOG_INFO("[TRACE] " << name() << " OUTPUT '" << tensor_name
                                << "' [no fp32 data available]");
            return;
        }

        std::ostringstream oss;
        oss << "[TRACE] " << name() << " OUTPUT '" << tensor_name << "' ";

        if (cfg.trace_shapes)
        {
            oss << "[" << tensor->rows() << "x" << tensor->cols() << " "
                << tensor->dtype_name() << "] ";
        }

        oss << formatFloatArray(data, count);

        if (cfg.trace_checksums)
        {
            oss << " checksum=" << computeChecksum(data, count);
        }

        LOG_INFO(oss.str());
    }

    void IComputeStage::traceIntermediate(const std::string &array_name, const float *data, size_t count) const
    {
        if (!shouldTrace() || !data)
            return;

        const auto &cfg = debugEnv().execution;

        std::ostringstream oss;
        oss << "[TRACE] " << name() << " INTERMEDIATE '" << array_name << "' ";

        if (cfg.trace_shapes)
        {
            oss << "[" << count << " elems] ";
        }

        oss << formatFloatArray(data, count);

        if (cfg.trace_checksums)
        {
            oss << " checksum=" << computeChecksum(data, count);
        }

        LOG_DEBUG(oss.str());
    }

    // =============================================================================
    // Shape Validation Implementation (Task 7)
    // =============================================================================

    void IComputeStage::validateShapes(std::initializer_list<TensorShapeContract> contracts) const
    {
        for (const auto &c : contracts)
        {
            if (!c.tensor)
            {
                std::ostringstream ss;
                ss << name() << ": null tensor '" << c.name << "'";
                throw std::runtime_error(ss.str());
            }

            const auto &actual = c.tensor->shape();
            if (!shapesMatch(actual, c.expected, c.allow_broadcast))
            {
                std::ostringstream ss;
                ss << name() << ": shape mismatch for '" << c.name << "' - "
                   << "expected " << shapeToString(c.expected)
                   << " but got " << shapeToString(actual);
                throw std::runtime_error(ss.str());
            }
        }
    }

    void IComputeStage::validateShape(const std::string &tensor_name, const TensorBase *tensor,
                                      const std::vector<size_t> &expected) const
    {
        validateShapes({{tensor_name, tensor, expected, false}});
    }

    void IComputeStage::validateMatmulShapes(const std::string &a_name, const TensorBase *a,
                                             const std::string &b_name, const TensorBase *b) const
    {
        if (!a || !b)
        {
            std::ostringstream ss;
            ss << name() << ": null tensor in matmul validation - "
               << a_name << "=" << (a ? "valid" : "null") << ", "
               << b_name << "=" << (b ? "valid" : "null");
            throw std::runtime_error(ss.str());
        }

        const auto &a_shape = a->shape();
        const auto &b_shape = b->shape();

        if (a_shape.size() < 2 || b_shape.size() < 2)
        {
            std::ostringstream ss;
            ss << name() << ": matmul requires 2D tensors - "
               << a_name << " has " << a_shape.size() << "D, "
               << b_name << " has " << b_shape.size() << "D";
            throw std::runtime_error(ss.str());
        }

        size_t a_k = a_shape[a_shape.size() - 1]; // Last dim of A
        size_t b_k = b_shape[b_shape.size() - 2]; // Second-to-last dim of B

        if (a_k != b_k)
        {
            std::ostringstream ss;
            ss << name() << ": matmul K dimension mismatch - "
               << a_name << " has K=" << a_k << " (shape " << shapeToString(a_shape) << "), "
               << b_name << " has K=" << b_k << " (shape " << shapeToString(b_shape) << ")";
            throw std::runtime_error(ss.str());
        }
    }

    bool IComputeStage::shapesMatch(const std::vector<size_t> &actual,
                                    const std::vector<size_t> &expected,
                                    bool allow_broadcast)
    {
        if (actual.size() != expected.size())
        {
            return false;
        }
        for (size_t i = 0; i < actual.size(); ++i)
        {
            if (actual[i] != expected[i])
            {
                // With broadcast, trailing dims of 1 match anything
                if (allow_broadcast && (actual[i] == 1 || expected[i] == 1))
                {
                    continue;
                }
                return false;
            }
        }
        return true;
    }

    std::string IComputeStage::shapeToString(const std::vector<size_t> &shape)
    {
        std::ostringstream ss;
        ss << "(";
        for (size_t i = 0; i < shape.size(); ++i)
        {
            if (i > 0)
                ss << ", ";
            ss << shape[i];
        }
        ss << ")";
        return ss.str();
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
    // GEMMStage::Params Implementation
    // =============================================================================

    const float *GEMMStage::Params::getBiasData() const
    {
        // Prefer bias_tensor over raw bias pointer for TensorSlice compatibility
        if (bias_tensor)
        {
            // Use unified data access interface - works for FP32Tensor and TensorSlice
            if (bias_tensor->is_fp32_backed())
            {
                return bias_tensor->data();
            }
            else
            {
                // Non-FP32-backed bias tensor (e.g., quantized) - not supported
                LOG_WARN("[GEMMStage::Params] bias_tensor is not FP32-backed, ignoring. "
                         "Type: "
                         << bias_tensor->dtype_name());
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
        if (params_.B && params_.B->shape()[0] == 896 && (params_.B->shape()[1] == 448 || params_.B->shape()[1] == 896))
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

        // Get kernel - either full or sliced
        llaminar2::ITensorGemm *gemm = nullptr;
        if (is_sliced)
        {
            // Use sliced kernel for tensor parallelism
            gemm = llaminar::v2::kernels::KernelFactory::getOrCreateGemmSliced(
                params_.B, params_.output_range.start, params_.output_range.end);
            LOG_DEBUG("[GEMMStage] Using sliced kernel for rows [" << params_.output_range.start
                                                                   << ", " << params_.output_range.end << ")");
        }
        else
        {
            // Use full kernel
            gemm = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(params_.B);
        }

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
                                                          << ", params: n=" << effective_n << " k=" << params_.k);
            if (kernel_n != effective_n || kernel_k != params_.k)
            {
                LOG_ERROR("[GEMMStage] DIMENSION MISMATCH! kernel N=" << kernel_n << " vs params n=" << effective_n
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
                bool success = qgemm->multiply_tensor(
                    params_.A, params_.C,
                    params_.m, effective_n, params_.k,
                    params_.transpose_B,
                    params_.alpha, params_.beta,
                    params_.mpi_ctx, params_.device_idx);

                // === Stage Tracing (Task 3) ===
                if (success)
                    traceOutput("C", params_.C);
                return success;
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
                params_.mpi_ctx, params_.device_idx,
                gate_fp32,
                params_.do_swiglu);

            // === Stage Tracing (Task 3) ===
            if (success)
                traceOutput("C", params_.C);
            return success;
        }

        // FP32 GEMM fallback (for non-quantized weights)
        // Use multiply_tensor if available for type-aware dispatch
        if (gemm->multiply_tensor(params_.A, params_.C, params_.m, effective_n, params_.k,
                                  params_.transpose_B, params_.alpha, params_.beta,
                                  params_.mpi_ctx, params_.device_idx))
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
            params_.alpha, params_.beta);

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
        info.addScalarInt("device_idx", params_.device_idx);

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
        LOG_TRACE("[FusedQKVGEMMStage] input_fp32 ptr=" << static_cast<const void *>(input_fp32)
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

        // Debug: Log Q output for comparison
        LOG_TRACE("[FusedQKVGEMMStage] Q output[0:8]=" << std::setprecision(10)
                                                       << output_q_fp32[0] << "," << output_q_fp32[1] << "," << output_q_fp32[2] << "," << output_q_fp32[3] << ","
                                                       << output_q_fp32[4] << "," << output_q_fp32[5] << "," << output_q_fp32[6] << "," << output_q_fp32[7]
                                                       << " n_q=" << params_.n_q);

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

    StageBufferRequirements FusedQKVGEMMStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (!params_.input || !params_.wq || !params_.wk || !params_.wv)
            return reqs; // Empty if tensors not set

        // Convert tensor types
        BufferTensorType input_type = params_.input
                                          ? toBufferTensorType(params_.input->native_type())
                                          : BufferTensorType::FP32;
        BufferTensorType wq_type = toBufferTensorType(params_.wq->native_type());
        BufferTensorType wk_type = toBufferTensorType(params_.wk->native_type());
        BufferTensorType wv_type = toBufferTensorType(params_.wv->native_type());

        // INPUT buffer (shared activation)
        reqs.addInput("input", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.k)}, input_type);

        // WEIGHT buffers
        reqs.addWeight("wq", {static_cast<size_t>(params_.k), static_cast<size_t>(params_.n_q)}, wq_type);
        reqs.addWeight("wk", {static_cast<size_t>(params_.k), static_cast<size_t>(params_.n_k)}, wk_type);
        reqs.addWeight("wv", {static_cast<size_t>(params_.k), static_cast<size_t>(params_.n_v)}, wv_type);

        // OUTPUT buffers
        BufferTensorType out_type = params_.output_q
                                        ? toBufferTensorType(params_.output_q->native_type())
                                        : BufferTensorType::FP32;
        reqs.addOutput("output_q", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.n_q)}, out_type);
        reqs.addOutput("output_k", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.n_k)}, out_type);
        reqs.addOutput("output_v", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.n_v)}, out_type);

        // Optional biases
        if (params_.bias_q)
        {
            reqs.addWeight("bias_q", {static_cast<size_t>(params_.n_q)}, BufferTensorType::FP32);
        }
        if (params_.bias_k)
        {
            reqs.addWeight("bias_k", {static_cast<size_t>(params_.n_k)}, BufferTensorType::FP32);
        }
        if (params_.bias_v)
        {
            reqs.addWeight("bias_v", {static_cast<size_t>(params_.n_v)}, BufferTensorType::FP32);
        }

        return reqs;
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

    StageBufferRequirements FusedGateUpGEMMStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (!params_.input || !params_.w_gate || !params_.w_up)
            return reqs; // Empty if tensors not set

        // Convert tensor types
        BufferTensorType input_type = toBufferTensorType(params_.input->native_type());
        BufferTensorType gate_type = toBufferTensorType(params_.w_gate->native_type());
        BufferTensorType up_type = toBufferTensorType(params_.w_up->native_type());

        // INPUT buffer (shared activation)
        reqs.addInput("input", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.k)}, input_type);

        // WEIGHT buffers
        reqs.addWeight("w_gate", {static_cast<size_t>(params_.k), static_cast<size_t>(params_.n_gate)}, gate_type);
        reqs.addWeight("w_up", {static_cast<size_t>(params_.k), static_cast<size_t>(params_.n_up)}, up_type);

        // OUTPUT buffers
        BufferTensorType out_type = params_.output_gate
                                        ? toBufferTensorType(params_.output_gate->native_type())
                                        : BufferTensorType::FP32;
        reqs.addOutput("output_gate", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.n_gate)}, out_type);
        reqs.addOutput("output_up", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.n_up)}, out_type);

        // Optional biases
        if (params_.bias_gate)
        {
            reqs.addWeight("bias_gate", {static_cast<size_t>(params_.n_gate)}, BufferTensorType::FP32);
        }
        if (params_.bias_up)
        {
            reqs.addWeight("bias_up", {static_cast<size_t>(params_.n_up)}, BufferTensorType::FP32);
        }

        return reqs;
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

        // === Stage Tracing (Task 3) ===
        traceInput("input", params_.input);
        traceInput("gamma", params_.gamma);

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
        bool success = kernel->apply_tensor(
            params_.input,
            params_.gamma,
            params_.output,
            seq_len,
            hidden_dim,
            params_.eps,
            params_.mpi_ctx,
            params_.device_idx);

        // === Stage Tracing (Task 3) ===
        if (success)
            traceOutput("output", params_.output);
        return success;
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

    StageBufferRequirements RMSNormStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (!params_.input)
            return reqs; // Empty if tensors not set

        // Get dimensions from tensors
        const size_t seq_len = params_.seq_len > 0
                                   ? static_cast<size_t>(params_.seq_len)
                                   : params_.input->rows();
        const size_t hidden_dim = params_.input->cols();

        // Convert tensor type to buffer tensor type
        BufferTensorType buf_type = toBufferTensorType(params_.input->native_type());

        // INPUT buffer (may be in-place with output)
        reqs.addInput("input", {seq_len, hidden_dim}, buf_type);

        // OUTPUT buffer
        if (params_.output)
        {
            reqs.addOutput("output", {seq_len, hidden_dim}, buf_type);
        }

        // WEIGHT buffer (gamma - always FP32)
        if (params_.gamma)
        {
            reqs.addWeight("gamma", {hidden_dim}, BufferTensorType::FP32);
        }

        return reqs;
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

        // Use explicit seq_len if provided, otherwise derive from tensor
        const int seq_len = (params_.seq_len > 0) ? params_.seq_len : static_cast<int>(params_.Q->rows());

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

    StageBufferRequirements RoPEStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (!params_.Q)
            return reqs; // Empty if tensors not set

        // Get dimensions from tensors
        const size_t seq_len = params_.Q->rows();
        const size_t q_dim = static_cast<size_t>(params_.n_heads * params_.head_dim);

        // Convert tensor type to buffer tensor type
        BufferTensorType buf_type = toBufferTensorType(params_.Q->native_type());

        // Q is INOUT (in-place operation)
        reqs.addInout("Q", {seq_len, q_dim}, buf_type);

        // K is optional INOUT (in-place operation)
        if (params_.K)
        {
            const int n_kv_heads = params_.n_kv_heads > 0 ? params_.n_kv_heads : params_.n_heads;
            const size_t k_dim = static_cast<size_t>(n_kv_heads * params_.head_dim);
            reqs.addInout("K", {seq_len, k_dim}, buf_type);
        }

        return reqs;
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

        // Use explicit seq_len if provided, otherwise derive from tensor shape
        // This is critical for decode mode where buffers are pre-allocated for max_seq_len
        // but we're only processing 1 token.
        const int seq_len = (params_.seq_len > 0)
                                ? params_.seq_len
                                : static_cast<int>(params_.gate->rows());
        const int intermediate_dim = static_cast<int>(params_.gate->cols());

        LOG_DEBUG("[SwiGLUStage] Execute: seq_len=" << seq_len
                                                    << " (params_.seq_len=" << params_.seq_len << ")"
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

        // Use explicit seq_len if provided, otherwise derive from tensor
        const int seq_len = (params_.seq_len > 0) ? params_.seq_len : static_cast<int>(params_.gate->rows());
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

    StageBufferRequirements SwiGLUStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (!params_.gate || !params_.up || !params_.output)
            return reqs; // Empty if tensors not set

        // Get dimensions from tensors
        const size_t rows = params_.gate->rows();
        const size_t cols = params_.gate->cols();

        // Convert tensor type to buffer tensor type
        BufferTensorType buf_type = toBufferTensorType(params_.gate->native_type());

        // INPUT buffers (read-only)
        reqs.addInput("gate", {rows, cols}, buf_type);
        reqs.addInput("up", {rows, cols}, buf_type);

        // OUTPUT buffer
        reqs.addOutput("output", {rows, cols}, buf_type);

        return reqs;
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

        // Use explicit num_elements if provided (for pre-allocated buffers)
        // Otherwise derive from tensor dimensions
        size_t actual_elements = (params_.num_elements > 0) ? params_.num_elements : params_.input->numel();
        int cols = static_cast<int>(params_.input->cols());
        int rows = (cols > 0) ? static_cast<int>(actual_elements / cols) : 1;

        // Input tensors - use safe FP32 accessor
        const float *input_data = getSafeFp32Data(params_.input);
        const float *residual_data = getSafeFp32Data(params_.residual);
        const float *output_data = getSafeFp32Data(params_.output);

        if (input_data)
        {
            info.inputs.push_back({"input", input_data,
                                   static_cast<size_t>(rows), static_cast<size_t>(cols),
                                   dtype, elem_size});
        }
        if (residual_data)
        {
            info.inputs.push_back({"residual", residual_data,
                                   static_cast<size_t>(rows), static_cast<size_t>(cols),
                                   dtype, elem_size});
        }

        // Output
        if (output_data)
        {
            info.outputs.push_back({"output", output_data,
                                    static_cast<size_t>(rows), static_cast<size_t>(cols),
                                    dtype, elem_size});
        }

        // Scalar params
        info.addScalarInt("num_elements", static_cast<int>(actual_elements));
        info.addScalarInt("rows", rows);
        info.addScalarInt("cols", cols);

        return info;
    }

    StageBufferRequirements ResidualAddStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (!params_.input || !params_.residual || !params_.output)
            return reqs; // Empty if tensors not set

        // Get dimensions from tensors
        const size_t rows = params_.input->rows();
        const size_t cols = params_.input->cols();

        // Convert tensor type to buffer tensor type
        BufferTensorType buf_type = toBufferTensorType(params_.input->native_type());

        // INPUT buffers (read-only)
        reqs.addInput("input", {rows, cols}, buf_type);
        reqs.addInput("residual", {rows, cols}, buf_type);

        // OUTPUT buffer (may alias residual for in-place operation)
        reqs.addOutput("output", {rows, cols}, buf_type);

        return reqs;
    }

    // =============================================================================
    // AllreduceStage Implementation
    // =============================================================================

    AllreduceStage::AllreduceStage(Params params) : params_(std::move(params)) {}

    bool AllreduceStage::execute(IDeviceContext *ctx)
    {
        (void)ctx;
        const auto &mpi_env = debugEnv().mpi_logging;

        if (!params_.buffer)
        {
            LOG_ERROR("[AllreduceStage] Null buffer");
            return false;
        }

        // Use explicit count if provided, otherwise use buffer size
        size_t count = params_.count > 0 ? params_.count : params_.buffer->numel();

        if (mpi_env.log_collectives)
        {
            LOG_INFO("[MPI] AllReduce START: count=" << count
                                                     << " dtype=" << params_.buffer->dtype_name());
        }

        LOG_DEBUG("[AllreduceStage] Execute: buffer=" << params_.buffer
                                                      << " count=" << count << " has_comm=" << (params_.mpi_comm != nullptr));
        if (!params_.mpi_comm)
        {
            LOG_ERROR("[AllreduceStage] Null MPI communicator");
            return false;
        }

        MPI_Comm comm = static_cast<MPI_Comm>(params_.mpi_comm);

        // Get mutable data pointer based on tensor type
        void *data_ptr = nullptr;
        MPI_Datatype mpi_type = MPI_FLOAT;

        if (params_.buffer->native_type() == TensorType::FP32)
        {
            auto *fp32_tensor = dynamic_cast<FP32Tensor *>(params_.buffer);
            if (fp32_tensor)
            {
                data_ptr = fp32_tensor->mutable_data();
                mpi_type = MPI_FLOAT;

                // Debug: dump values before AllReduce
                float *f = static_cast<float *>(data_ptr);
                LOG_DEBUG("[AllreduceStage] Before AllReduce: data[0:4]="
                          << f[0] << "," << f[1] << "," << f[2] << "," << f[3]);
            }
        }
        // Add other types as needed (BF16, FP16, etc.)

        if (!data_ptr)
        {
            LOG_ERROR("[AllreduceStage] Unsupported tensor type for allreduce");
            return false;
        }

        // Compute checksum before if verification enabled
        float checksum_before = 0.0f;
        if (mpi_env.verify_checksums && mpi_type == MPI_FLOAT)
        {
            const float *f = static_cast<const float *>(data_ptr);
            for (size_t i = 0; i < count; ++i)
            {
                checksum_before += f[i];
            }
            LOG_INFO("[MPI] AllReduce checksum BEFORE: " << checksum_before);
        }

        LOG_DEBUG("[AllreduceStage] Calling MPI_Allreduce with count=" << count);

        // Start timing if enabled
        auto start_time = std::chrono::high_resolution_clock::now();

        int result = MPI_Allreduce(
            MPI_IN_PLACE,
            data_ptr,
            static_cast<int>(count),
            mpi_type,
            MPI_SUM,
            comm);

        // End timing
        auto end_time = std::chrono::high_resolution_clock::now();

        // Debug: dump values after AllReduce
        if (mpi_type == MPI_FLOAT)
        {
            float *f = static_cast<float *>(data_ptr);
            LOG_DEBUG("[AllreduceStage] After AllReduce: data[0:4]="
                      << f[0] << "," << f[1] << "," << f[2] << "," << f[3]);
        }

        // Log timing if enabled
        if (mpi_env.log_timing)
        {
            double ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            double bytes = count * sizeof(float);
            double bandwidth_gbps = (bytes / (ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
            LOG_INFO("[MPI] AllReduce timing: " << ms << " ms for " << count
                                                << " elements (" << bandwidth_gbps << " GB/s)");
        }

        // Verify checksum after if enabled
        if (mpi_env.verify_checksums && mpi_type == MPI_FLOAT)
        {
            float checksum_after = 0.0f;
            const float *f = static_cast<const float *>(data_ptr);
            for (size_t i = 0; i < count; ++i)
            {
                checksum_after += f[i];
            }
            LOG_INFO("[MPI] AllReduce checksum AFTER: " << checksum_after);
        }

        if (mpi_env.log_collectives)
        {
            LOG_INFO("[MPI] AllReduce END: result=" << (result == MPI_SUCCESS ? "SUCCESS" : "FAILED"));
        }

        LOG_DEBUG("[AllreduceStage] MPI_Allreduce returned " << result);
        return result == MPI_SUCCESS;
    }

    bool AllreduceStage::supportsBackend(ComputeBackendType backend) const
    {
        // Allreduce is backend-agnostic (works with any device that has MPI support)
        (void)backend;
        return true;
    }

    StageBufferRequirements AllreduceStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        // Allreduce operates in-place on a single buffer
        if (params_.buffer)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.buffer->native_type());
            reqs.addInout("buffer", params_.buffer->shape(), buf_type);
        }

        return reqs;
    }

    // =============================================================================
    // AllGatherStage Implementation
    // =============================================================================

    AllGatherStage::AllGatherStage(Params params) : params_(std::move(params)) {}

    bool AllGatherStage::execute(IDeviceContext *ctx)
    {
        (void)ctx;
        const auto &mpi_env = debugEnv().mpi_logging;

        if (!params_.local_input)
        {
            LOG_ERROR("[AllGatherStage] Null local_input buffer");
            return false;
        }

        if (!params_.full_output)
        {
            LOG_ERROR("[AllGatherStage] Null full_output buffer");
            return false;
        }

        if (!params_.mpi_comm)
        {
            LOG_ERROR("[AllGatherStage] Null MPI communicator");
            return false;
        }

        if (params_.world_size <= 0)
        {
            LOG_ERROR("[AllGatherStage] Invalid world_size=" << params_.world_size);
            return false;
        }

        // Get shapes: local_input = [seq_len, vocab_local], full_output = [seq_len, vocab_size]
        const auto &local_shape = params_.local_input->shape();
        const auto &full_shape = params_.full_output->shape();

        if (local_shape.size() != 2 || full_shape.size() != 2)
        {
            LOG_ERROR("[AllGatherStage] Expected 2D tensors, got local_shape.size()="
                      << local_shape.size() << " full_shape.size()=" << full_shape.size());
            return false;
        }

        // Use actual_seq_len if provided, otherwise use buffer shape
        size_t buffer_seq_len = local_shape[0];
        size_t seq_len = params_.actual_seq_len > 0 ? params_.actual_seq_len : buffer_seq_len;
        size_t vocab_local = local_shape[1];
        size_t vocab_full = full_shape[0] == buffer_seq_len ? full_shape[1] : full_shape[0];

        // Validate seq_len doesn't exceed buffer capacity
        if (seq_len > buffer_seq_len)
        {
            LOG_ERROR("[AllGatherStage] actual_seq_len=" << seq_len
                                                         << " exceeds buffer capacity=" << buffer_seq_len);
            return false;
        }

        size_t expected_vocab_full = vocab_local * static_cast<size_t>(params_.world_size);
        if (vocab_full != expected_vocab_full)
        {
            LOG_WARN("[AllGatherStage] vocab_full=" << vocab_full
                                                    << " expected=" << expected_vocab_full
                                                    << " - proceeding anyway");
        }

        LOG_DEBUG("[AllGatherStage] Execute: seq_len=" << seq_len
                                                       << " vocab_local=" << vocab_local
                                                       << " vocab_full=" << vocab_full
                                                       << " world_size=" << params_.world_size);

        MPI_Comm comm = static_cast<MPI_Comm>(params_.mpi_comm);

        // Get data pointers based on tensor type
        const float *send_ptr = nullptr;
        float *recv_ptr = nullptr;

        if (params_.local_input->native_type() == TensorType::FP32)
        {
            auto *input_fp32 = dynamic_cast<FP32Tensor *>(params_.local_input);
            auto *output_fp32 = dynamic_cast<FP32Tensor *>(params_.full_output);
            if (input_fp32 && output_fp32)
            {
                send_ptr = input_fp32->data();
                recv_ptr = output_fp32->mutable_data();
            }
        }

        if (!send_ptr || !recv_ptr)
        {
            LOG_ERROR("[AllGatherStage] Unsupported tensor type for allgather");
            return false;
        }

        // =========================================================================
        // Optimized AllGather using MPI_Type_vector for strided data
        // =========================================================================
        // Each rank has [seq_len, vocab_local] data laid out contiguously.
        // We need output [seq_len, vocab_full] where each row interleaves all ranks' data:
        //   row_i: [rank0's vocab_local elements, rank1's vocab_local elements, ...]
        //
        // Strategy: Use MPI_Type_vector to describe strided placement in output buffer.
        // - Send type: seq_len blocks of vocab_local, stride vocab_local (contiguous)
        // - Recv type: seq_len blocks of vocab_local, stride vocab_full (strided)
        //
        // MPI_Allgather with derived types places each rank's recv_type starting at:
        //   rank * recvcount * sizeof(recv_type_extent)
        // But we need placement at: rank * vocab_local elements within each row.
        //
        // Solution: Create a resized recv type where the extent equals vocab_local elements,
        // so rank r's data lands at offset r * vocab_local within the vocab dimension.
        // =========================================================================

        // Step 1: Create the basic strided vector type
        // seq_len blocks of vocab_local floats, each block starting vocab_full floats apart
        MPI_Datatype strided_type;
        int mpi_result = MPI_Type_vector(
            static_cast<int>(seq_len),     // count: number of blocks (rows)
            static_cast<int>(vocab_local), // blocklength: elements per block
            static_cast<int>(vocab_full),  // stride: distance between block starts (in elements)
            MPI_FLOAT,
            &strided_type);

        if (mpi_result != MPI_SUCCESS)
        {
            LOG_ERROR("[AllGatherStage] MPI_Type_vector failed with error " << mpi_result);
            return false;
        }

        // Step 2: Resize the type so its extent = vocab_local * sizeof(float)
        // This ensures consecutive ranks' data are placed vocab_local apart in each row
        MPI_Datatype resized_type;
        MPI_Aint lb = 0;
        MPI_Aint extent = static_cast<MPI_Aint>(vocab_local) * sizeof(float);

        mpi_result = MPI_Type_create_resized(strided_type, lb, extent, &resized_type);
        MPI_Type_free(&strided_type);

        if (mpi_result != MPI_SUCCESS)
        {
            LOG_ERROR("[AllGatherStage] MPI_Type_create_resized failed with error " << mpi_result);
            return false;
        }

        mpi_result = MPI_Type_commit(&resized_type);
        if (mpi_result != MPI_SUCCESS)
        {
            LOG_ERROR("[AllGatherStage] MPI_Type_commit failed with error " << mpi_result);
            MPI_Type_free(&resized_type);
            return false;
        }

        // Log MPI collective start
        if (mpi_env.log_collectives)
        {
            LOG_INFO("[MPI] AllGather START: seq_len=" << seq_len
                                                       << " vocab_local=" << vocab_local
                                                       << " vocab_full=" << vocab_full
                                                       << " world_size=" << params_.world_size);
        }

        // Start timing if enabled
        auto start_time = std::chrono::high_resolution_clock::now();

        // Step 3: Use MPI_Allgather with the resized type
        // Send: seq_len * vocab_local contiguous floats
        // Recv: 1 resized_type per rank (each type covers seq_len strided blocks)
        mpi_result = MPI_Allgather(
            send_ptr,
            static_cast<int>(seq_len * vocab_local),
            MPI_FLOAT,
            recv_ptr,
            1,
            resized_type,
            comm);

        // End timing
        auto end_time = std::chrono::high_resolution_clock::now();

        // Clean up the derived type
        MPI_Type_free(&resized_type);

        if (mpi_result != MPI_SUCCESS)
        {
            LOG_ERROR("[AllGatherStage] MPI_Allgather (strided) failed with error " << mpi_result);
            return false;
        }

        // Log timing if enabled
        if (mpi_env.log_timing)
        {
            double ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            size_t total_elements = seq_len * vocab_local;
            double bytes = total_elements * sizeof(float) * params_.world_size; // Total bytes gathered
            double bandwidth_gbps = (bytes / (ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
            LOG_INFO("[MPI] AllGather timing: " << ms << " ms for " << total_elements
                                                << " elements/rank (" << bandwidth_gbps << " GB/s aggregate)");
        }

        if (mpi_env.log_collectives)
        {
            LOG_INFO("[MPI] AllGather END: result=SUCCESS");
        }

        // Debug: dump gathered logits at last position
        LOG_DEBUG("[AllGatherStage] Gathered logits (last row, first 8): "
                  << recv_ptr[(seq_len - 1) * vocab_full + 0] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + 1] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + 2] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + 3] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + 4] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + 5] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + 6] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + 7]);
        // Also dump from second half (rank 1's data)
        LOG_DEBUG("[AllGatherStage] Gathered logits (last row, at vocab_local+0:+8): "
                  << recv_ptr[(seq_len - 1) * vocab_full + vocab_local + 0] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + vocab_local + 1] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + vocab_local + 2] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + vocab_local + 3] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + vocab_local + 4] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + vocab_local + 5] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + vocab_local + 6] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + vocab_local + 7]);

        LOG_DEBUG("[AllGatherStage] Optimized AllGather (MPI_Type_vector) completed for "
                  << seq_len << " rows");
        return true;
    }

    bool AllGatherStage::supportsBackend(ComputeBackendType backend) const
    {
        // AllGather is backend-agnostic (works with any device that has MPI support)
        (void)backend;
        return true;
    }

    StageBufferRequirements AllGatherStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        // AllGather has separate input and output buffers
        if (params_.local_input)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.local_input->native_type());
            reqs.addInput("local_input", params_.local_input->shape(), buf_type);
        }

        if (params_.full_output)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.full_output->native_type());
            reqs.addOutput("full_output", params_.full_output->shape(), buf_type);
        }

        return reqs;
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

        if (!params_.hidden || !params_.gate_weights || !params_.router_logits)
        {
            LOG_ERROR("[MoERouterStage] Null tensor parameters");
            return false;
        }

        // Get data pointers from tensors
        const float *hidden = params_.hidden->data();
        const float *gate_weights = params_.gate_weights->data();

        // Router logits is output, need mutable access
        auto *logits_tensor = dynamic_cast<FP32Tensor *>(params_.router_logits);
        if (!logits_tensor)
        {
            LOG_ERROR("[MoERouterStage] router_logits must be FP32Tensor");
            return false;
        }
        float *logits = logits_tensor->mutable_data();

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

    StageBufferRequirements MoERouterStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (params_.hidden)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.hidden->native_type());
            reqs.addInput("hidden", params_.hidden->shape(), buf_type);
        }

        if (params_.gate_weights)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.gate_weights->native_type());
            reqs.addWeight("gate_weights", params_.gate_weights->shape(), buf_type);
        }

        if (params_.router_logits)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.router_logits->native_type());
            reqs.addOutput("router_logits", params_.router_logits->shape(), buf_type);
        }

        return reqs;
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

        if (!params_.input || !params_.output)
        {
            LOG_ERROR("[MoEExpertStage] Null input or output tensor");
            return false;
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

    StageBufferRequirements MoEExpertStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (params_.input)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.input->native_type());
            reqs.addInput("input", params_.input->shape(), buf_type);
        }

        if (params_.output)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.output->native_type());
            reqs.addOutput("output", params_.output->shape(), buf_type);
        }

        // Note: Expert weights (gate, up, down) would be added here when we have
        // proper weight tensor references in the Params struct

        return reqs;
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

    StageBufferRequirements MoECombineStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        // Add expert outputs as inputs
        if (params_.expert_outputs)
        {
            for (size_t i = 0; i < params_.expert_outputs->size(); ++i)
            {
                const TensorBase *expert_out = (*params_.expert_outputs)[i];
                if (expert_out)
                {
                    BufferTensorType buf_type = toBufferTensorType(expert_out->native_type());
                    std::string name = "expert_output_" + std::to_string(i);
                    reqs.addInput(name, expert_out->shape(), buf_type);
                }
            }
        }

        if (params_.output)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.output->native_type());
            reqs.addOutput("output", params_.output->shape(), buf_type);
        }

        return reqs;
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

        // Step 4: Create attention kernel via KernelFactory (device-aware dispatch)
        const int device_idx = params_.device_idx;
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_idx);
        auto attention_kernel = llaminar::v2::kernels::KernelFactory::createAttention(Q_view.get(), dev_type);

        if (!attention_kernel)
        {
            LOG_ERROR("[AttentionWithKVCacheStage::executePrefill] Failed to create attention kernel");
            return false;
        }

        // Step 5: Allocate workspace for attention scores [n_heads, seq_len, kv_len]
        FP32Tensor scores_workspace({static_cast<size_t>(params_.n_heads * params_.seq_len * kv_len)});

        // Step 6: Build causal mask if needed
        std::unique_ptr<FP32Tensor> mask_tensor;
        if (params_.causal)
        {
            mask_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(params_.seq_len * kv_len)});
            float *mask_data = mask_tensor->mutable_data();
            for (int q = 0; q < params_.seq_len; ++q)
            {
                for (int k = 0; k < kv_len; ++k)
                {
                    // Causal: Q at position q can attend to K at positions [0, q]
                    mask_data[q * kv_len + k] = (k <= q) ? 0.0f : -std::numeric_limits<float>::infinity();
                }
            }
        }

        // Step 7: Dispatch via compute_tensor
        bool success = attention_kernel->compute_tensor(
            Q_view.get(), K_view.get(), V_view.get(), out_view.get(),
            params_.batch_size,
            params_.seq_len, // Query sequence length
            kv_len,          // Key/value sequence length
            params_.n_heads,
            params_.n_kv_heads,
            params_.head_dim,
            params_.causal,
            params_.window_size,
            &scores_workspace,
            mask_tensor.get(),
            params_.mpi_ctx.get(),
            device_idx,
            params_.head_start,
            params_.local_n_heads,
            params_.local_n_kv_heads);

        if (!success)
        {
            LOG_ERROR("[AttentionWithKVCacheStage::executePrefill] Attention kernel compute_tensor failed");
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
        // This uses the typed kernel path (CPUAttentionKernelT) which properly
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
                                                                       << " batch_size=" << params_.batch_size << " seq_len=" << params_.seq_len
                                                                       << " sequence_lengths=" << (params_.sequence_lengths ? "valid" : "nullptr")
                                                                       << (params_.sequence_lengths ? " [0]=" + std::to_string((*params_.sequence_lengths)[0]) : ""));

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

        // Create attention kernel via KernelFactory (device-aware dispatch)
        const int device_idx = params_.device_idx;
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_idx);
        auto attention_kernel = llaminar::v2::kernels::KernelFactory::createAttention(params_.Q, dev_type);

        if (!attention_kernel)
        {
            LOG_ERROR("[AttentionWithKVCacheStage::executeBatched] Failed to create attention kernel");
            return false;
        }

        // Allocate workspace for attention scores
        // For batched: [batch_size * n_heads, seq_len, seq_len] but we compute per-batch
        FP32Tensor scores_workspace({static_cast<size_t>(params_.n_heads * params_.seq_len * params_.seq_len)});

        // Build padding-aware causal mask if needed
        std::unique_ptr<FP32Tensor> mask_tensor;
        if (params_.causal || params_.sequence_lengths)
        {
            // For batched: need full [batch*seq, batch*seq] mask for cross-batch isolation
            const int total_tokens = params_.batch_size * params_.seq_len;
            mask_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(total_tokens * total_tokens)});
            float *mask_data = mask_tensor->mutable_data();

            // Initialize with -inf (block all)
            std::fill(mask_data, mask_data + total_tokens * total_tokens, -std::numeric_limits<float>::infinity());

            // Allow within-batch attention with causal masking
            for (int b = 0; b < params_.batch_size; ++b)
            {
                int actual_len = params_.sequence_lengths ? (*params_.sequence_lengths)[b] : params_.seq_len;
                int batch_offset = b * params_.seq_len;

                for (int q = 0; q < actual_len; ++q)
                {
                    int q_pos = batch_offset + q;
                    for (int k = 0; k < actual_len; ++k)
                    {
                        int k_pos = batch_offset + k;
                        // Causal: Q at position q can attend to K at positions [0, q] within batch
                        if (!params_.causal || k <= q)
                        {
                            mask_data[q_pos * total_tokens + k_pos] = 0.0f;
                        }
                    }
                }
            }
        }

        // Dispatch via compute_tensor
        bool success = attention_kernel->compute_tensor(
            params_.Q, params_.K, params_.V, params_.output,
            params_.batch_size,
            params_.seq_len, // Query sequence length
            params_.seq_len, // KV length (same for batched prefill)
            params_.n_heads,
            params_.n_kv_heads,
            params_.head_dim,
            params_.causal,
            params_.window_size,
            &scores_workspace,
            mask_tensor.get(),
            params_.mpi_ctx.get(),
            device_idx,
            params_.head_start,
            params_.local_n_heads,
            params_.local_n_kv_heads);

        if (!success)
        {
            LOG_ERROR("[AttentionWithKVCacheStage::executeBatched] Attention kernel compute_tensor failed");
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

    StageBufferRequirements AttentionWithKVCacheStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        // Input: Q (query)
        if (params_.Q)
        {
            const size_t seq_dim = static_cast<size_t>(params_.batch_size * params_.seq_len);
            const size_t q_dim = static_cast<size_t>(params_.n_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.Q->native_type());
            reqs.addInput("Q", {seq_dim, q_dim}, buf_type);
        }

        // Input: K (key)
        if (params_.K)
        {
            const size_t seq_dim = static_cast<size_t>(params_.batch_size * params_.seq_len);
            const size_t k_dim = static_cast<size_t>(params_.n_kv_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.K->native_type());
            reqs.addInput("K", {seq_dim, k_dim}, buf_type);
        }

        // Input: V (value)
        if (params_.V)
        {
            const size_t seq_dim = static_cast<size_t>(params_.batch_size * params_.seq_len);
            const size_t v_dim = static_cast<size_t>(params_.n_kv_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.V->native_type());
            reqs.addInput("V", {seq_dim, v_dim}, buf_type);
        }

        // Output: attention output
        if (params_.output)
        {
            const size_t seq_dim = static_cast<size_t>(params_.batch_size * params_.seq_len);
            const size_t out_dim = static_cast<size_t>(params_.n_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.output->native_type());
            reqs.addOutput("output", {seq_dim, out_dim}, buf_type);
        }

        // Scratch: workspace buffers (if pre-allocated)
        if (params_.workspace_scores)
        {
            reqs.addScratch("workspace_scores", params_.workspace_scores->shape(),
                            toBufferTensorType(params_.workspace_scores->native_type()));
        }
        if (params_.workspace_context)
        {
            reqs.addScratch("workspace_context", params_.workspace_context->shape(),
                            toBufferTensorType(params_.workspace_context->native_type()));
        }

        return reqs;
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

        // Determine total tokens to append
        int total_tokens = params_.num_tokens;
        if (total_tokens <= 0)
        {
            total_tokens = static_cast<int>(params_.K->shape()[0]);
        }

        // Determine batch handling mode
        const int batch_size = params_.batch_size;
        const int seq_len = params_.seq_len;

        // If batch_size > 1 and seq_len > 0, do per-sequence append
        // K/V layout: [batch_size * seq_len, kv_dim] - contiguous per-sequence
        if (batch_size > 1 && seq_len > 0)
        {
            const size_t kv_dim = params_.K->shape().size() > 1 ? params_.K->shape()[1] : 0;

            LOG_DEBUG("[KVCacheAppendStage] Batched append: batch_size=" << batch_size
                                                                         << " seq_len=" << seq_len
                                                                         << " kv_dim=" << kv_dim
                                                                         << " layer=" << params_.layer_idx);

            // Get raw data pointers for slicing
            const float *k_data = params_.K->fp32_data();
            const float *v_data = params_.V->fp32_data();

            if (!k_data || !v_data)
            {
                LOG_ERROR("[KVCacheAppendStage] Cannot get FP32 data for K/V tensors");
                return false;
            }

            // Create temporary tensors for per-sequence slices
            // Note: We create views/copies that the cache will copy from
            for (int b = 0; b < batch_size; ++b)
            {
                const int seq_idx = params_.seq_idx + b;
                const size_t offset = b * seq_len * kv_dim;

                // Create temporary FP32 tensors wrapping the slice data
                // These are views into the contiguous K/V buffer
                auto k_slice = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(seq_len), kv_dim});
                auto v_slice = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(seq_len), kv_dim});

                // Copy slice data (could be optimized with non-owning views)
                std::memcpy(k_slice->mutable_data(), k_data + offset, seq_len * kv_dim * sizeof(float));
                std::memcpy(v_slice->mutable_data(), v_data + offset, seq_len * kv_dim * sizeof(float));

                LOG_TRACE("[KVCacheAppendStage] Appending " << seq_len << " tokens to layer "
                                                            << params_.layer_idx << " seq_idx=" << seq_idx);

                bool success = params_.kv_cache->append_kv(
                    params_.layer_idx, seq_idx,
                    k_slice.get(), v_slice.get(), seq_len);

                if (!success)
                {
                    LOG_ERROR("[KVCacheAppendStage] append_kv failed for batch " << b);
                    return false;
                }
            }

            return true;
        }

        // Single-sequence path (original behavior)
        LOG_DEBUG("[KVCacheAppendStage] Single-sequence append: " << total_tokens
                                                                  << " tokens to layer " << params_.layer_idx << " seq " << params_.seq_idx);

        bool success = params_.kv_cache->append_kv(
            params_.layer_idx, params_.seq_idx,
            params_.K, params_.V, total_tokens);

        if (!success)
        {
            LOG_ERROR("[KVCacheAppendStage] append_kv failed");
            return false;
        }

        return true;
    }

    StageBufferRequirements KVCacheAppendStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        // Input: K (to be appended to cache)
        if (params_.K)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.K->native_type());
            reqs.addInput("K", params_.K->shape(), buf_type);
        }

        // Input: V (to be appended to cache)
        if (params_.V)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.V->native_type());
            reqs.addInput("V", params_.V->shape(), buf_type);
        }

        // Note: KV cache itself is external state, not a buffer managed by this stage

        return reqs;
    }

    // =============================================================================
    // KVCacheGatherStage Implementation
    // =============================================================================

    KVCacheGatherStage::KVCacheGatherStage(Params params)
        : params_(std::move(params)) {}

    bool KVCacheGatherStage::execute(IDeviceContext *ctx)
    {
        (void)ctx;

        if (!params_.kv_cache)
        {
            LOG_ERROR("[KVCacheGatherStage] No KV cache provided");
            return false;
        }

        if (!params_.out_K || !params_.out_V)
        {
            LOG_ERROR("[KVCacheGatherStage] Invalid output K/V tensors");
            return false;
        }

        if (params_.batch_size <= 0)
        {
            LOG_ERROR("[KVCacheGatherStage] Invalid batch_size=" << params_.batch_size);
            return false;
        }

        // Call the unified gather method
        int max_kv_len = params_.kv_cache->gather_kv_batched(
            params_.layer_idx,
            params_.batch_size,
            params_.out_K,
            params_.out_V,
            last_per_seq_kv_lens_);

        if (max_kv_len < 0)
        {
            LOG_ERROR("[KVCacheGatherStage] gather_kv_batched failed for layer " << params_.layer_idx);
            return false;
        }

        last_max_kv_len_ = max_kv_len;

        // Write outputs if requested
        if (params_.out_max_kv_len)
        {
            *params_.out_max_kv_len = max_kv_len;
        }
        if (params_.out_per_seq_kv_lens)
        {
            *params_.out_per_seq_kv_lens = last_per_seq_kv_lens_;
        }

        LOG_DEBUG("[KVCacheGatherStage] Gathered " << params_.batch_size
                                                   << " sequences from layer " << params_.layer_idx
                                                   << ", max_kv_len=" << max_kv_len);

        return true;
    }

    StageBufferRequirements KVCacheGatherStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        // Output: K (gathered from cache)
        if (params_.out_K)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.out_K->native_type());
            reqs.addOutput("gathered_K", params_.out_K->shape(), buf_type);
        }

        // Output: V (gathered from cache)
        if (params_.out_V)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.out_V->native_type());
            reqs.addOutput("gathered_V", params_.out_V->shape(), buf_type);
        }

        // Note: KV cache itself is external state, not a buffer managed by this stage

        return reqs;
    }

    // =============================================================================
    // AttentionComputeStage Implementation
    // =============================================================================

    AttentionComputeStage::AttentionComputeStage(Params params)
        : params_(std::move(params)) {}

    bool AttentionComputeStage::execute(IDeviceContext *ctx)
    {
        // Dynamic kv_len: query from KV cache at execution time if available
        // This enables declarative graph construction where the stage runs after
        // KVCacheAppendStage has already appended tokens
        int effective_kv_len = params_.kv_len;
        if (params_.kv_cache && params_.layer_idx >= 0)
        {
            effective_kv_len = params_.kv_cache->get_cached_tokens(params_.layer_idx, 0);
            if (effective_kv_len == 0)
            {
                effective_kv_len = params_.seq_len; // Prefill case
            }
            LOG_TRACE("[AttentionComputeStage] Dynamic kv_len from cache: " << effective_kv_len
                                                                            << " (static was: " << params_.kv_len << ")");
        }

        // Detect attention mode if auto-detection enabled
        AttentionMode mode = params_.attention_mode;
        if (params_.auto_detect_mode)
        {
            mode = detect_attention_mode(params_.batch_size, params_.seq_len, effective_kv_len);
        }

        LOG_DEBUG("[AttentionComputeStage] Execute: batch=" << params_.batch_size
                                                            << " seq_len=" << params_.seq_len
                                                            << " kv_len=" << effective_kv_len
                                                            << " n_heads=" << params_.n_heads
                                                            << " n_kv_heads=" << params_.n_kv_heads
                                                            << " head_dim=" << params_.head_dim
                                                            << " position_offset=" << params_.position_offset
                                                            << " mode=" << attention_mode_name(mode));

        // Validate inputs
        if (!params_.Q || !params_.K || !params_.V || !params_.output)
        {
            LOG_ERROR("[AttentionComputeStage] Null tensor pointers");
            return false;
        }

        if (params_.seq_len <= 0 || effective_kv_len <= 0 ||
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

        // Build proper causal mask for decode mode
        // In decode mode (seq_len < kv_len), we need to account for position offset
        // The kernel's internal causal mask assumes m=0 for decode (query position 0),
        // but decode tokens should be able to attend to all cached positions [0, kv_len-1]
        //
        // Key insight: For decode with seq_len=1, the query is at position (kv_len-1),
        // so it should attend to ALL kv_len positions. The kernel's "n > m" check would
        // only allow attending to position 0, which is wrong.
        std::unique_ptr<FP32Tensor> decode_mask;
        TensorBase *mask_to_use = params_.workspace_mask;

        const bool is_decode_mode = (mode == AttentionMode::DECODE ||
                                     (params_.seq_len < effective_kv_len && params_.batch_size == 1));

        if (params_.causal && is_decode_mode)
        {
            // Build decode-specific causal mask
            // For decode: seq_len=1 (or small), kv_len = full cache length
            // Query at position i (within seq_len) corresponds to global position (base_pos + i)
            // where base_pos = position_offset if provided, else (kv_len - seq_len)
            const int base_pos = (params_.position_offset > 0)
                                     ? params_.position_offset
                                     : (effective_kv_len - params_.seq_len);

            decode_mask = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(params_.seq_len * effective_kv_len)});
            float *mask_data = decode_mask->mutable_data();

            for (int q = 0; q < params_.seq_len; ++q)
            {
                const int q_pos = base_pos + q; // Global position of this query
                for (int k = 0; k < effective_kv_len; ++k)
                {
                    // Causal: Query at position q_pos can attend to K positions [0, q_pos]
                    mask_data[q * effective_kv_len + k] = (k <= q_pos)
                                                              ? 0.0f
                                                              : -std::numeric_limits<float>::infinity();
                }
            }

            mask_to_use = decode_mask.get();
            LOG_DEBUG("[AttentionComputeStage] Built decode causal mask: base_pos=" << base_pos
                                                                                    << " seq_len=" << params_.seq_len << " kv_len=" << effective_kv_len);
        }

        // Dispatch to kernel's compute_tensor() method with detected mode
        // IMPORTANT: For decode with explicit mask, we pass causal=false to avoid
        // double-masking (kernel would apply "n > m" on top of our mask)
        const bool kernel_causal = params_.causal && !is_decode_mode;

        bool success = kernel->compute_tensor(
            params_.Q, params_.K, params_.V, params_.output,
            params_.batch_size,
            params_.seq_len,
            effective_kv_len,
            params_.n_heads,
            params_.n_kv_heads,
            params_.head_dim,
            kernel_causal, // Pass false for decode (we built the mask explicitly)
            params_.window_size,
            params_.workspace_scores,
            mask_to_use, // Use our decode mask if we built one
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

        // Output: attention context
        // Output shape: [batch_size * seq_len, n_heads * head_dim]
        if (params_.output)
        {
            const float *out_data = getSafeFp32Data(params_.output);
            if (out_data)
            {
                info.addOutput("output",
                               out_data,
                               params_.seq_len,
                               params_.n_heads * params_.head_dim);
            }
        }

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

    StageBufferRequirements AttentionComputeStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        // Input: Q (query)
        if (params_.Q)
        {
            const size_t q_rows = static_cast<size_t>(params_.batch_size * params_.seq_len);
            const size_t q_cols = static_cast<size_t>(params_.n_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.Q->native_type());
            reqs.addInput("Q", {q_rows, q_cols}, buf_type);
        }

        // Input: K (key - may have different kv_len than Q's seq_len)
        if (params_.K)
        {
            const size_t k_rows = static_cast<size_t>(params_.batch_size * params_.kv_len);
            const size_t k_cols = static_cast<size_t>(params_.n_kv_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.K->native_type());
            reqs.addInput("K", {k_rows, k_cols}, buf_type);
        }

        // Input: V (value)
        if (params_.V)
        {
            const size_t v_rows = static_cast<size_t>(params_.batch_size * params_.kv_len);
            const size_t v_cols = static_cast<size_t>(params_.n_kv_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.V->native_type());
            reqs.addInput("V", {v_rows, v_cols}, buf_type);
        }

        // Output: attention output
        if (params_.output)
        {
            const size_t out_rows = static_cast<size_t>(params_.batch_size * params_.seq_len);
            const size_t out_cols = static_cast<size_t>(params_.n_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.output->native_type());
            reqs.addOutput("output", {out_rows, out_cols}, buf_type);
        }

        // Scratch: workspace buffers (if pre-allocated)
        if (params_.workspace_scores)
        {
            reqs.addScratch("workspace_scores", params_.workspace_scores->shape(),
                            toBufferTensorType(params_.workspace_scores->native_type()));
        }
        if (params_.workspace_context)
        {
            reqs.addScratch("workspace_context", params_.workspace_context->shape(),
                            toBufferTensorType(params_.workspace_context->native_type()));
        }

        return reqs;
    }

    // =============================================================================
    // FusedAttentionWoStage Implementation
    // =============================================================================

    FusedAttentionWoStage::FusedAttentionWoStage(Params params)
        : params_(std::move(params))
    {
        // Create the kernel with configuration
        FusedAttentionWoKernel::Config kernel_config;
        kernel_config.num_heads = params_.n_heads;
        kernel_config.num_kv_heads = params_.n_kv_heads;
        kernel_config.head_dim = params_.head_dim;
        kernel_config.d_model = params_.d_model;
        kernel_config.backend = params_.backend;

        kernel_ = std::make_unique<FusedAttentionWoKernel>(kernel_config);
    }

    bool FusedAttentionWoStage::execute(IDeviceContext *ctx)
    {
        // Dynamic kv_len: query from KV cache at execution time if available
        int effective_kv_len = params_.kv_len;
        if (params_.kv_cache && params_.layer_idx >= 0)
        {
            effective_kv_len = params_.kv_cache->get_cached_tokens(params_.layer_idx, 0);
            if (effective_kv_len == 0)
            {
                effective_kv_len = params_.seq_len; // Prefill case
            }
            LOG_TRACE("[FusedAttentionWoStage] Dynamic kv_len from cache: " << effective_kv_len
                                                                            << " (static was: " << params_.kv_len << ")");
        }

        LOG_DEBUG("[FusedAttentionWoStage] Execute: batch=" << params_.batch_size
                                                            << " seq_len=" << params_.seq_len
                                                            << " kv_len=" << effective_kv_len
                                                            << " n_heads=" << params_.n_heads
                                                            << " n_kv_heads=" << params_.n_kv_heads
                                                            << " head_dim=" << params_.head_dim
                                                            << " d_model=" << params_.d_model
                                                            << " position_offset=" << params_.position_offset
                                                            << " backend=" << fusedAttentionBackendToString(params_.backend));

        // Validate inputs
        if (!params_.Q || !params_.K || !params_.V || !params_.Wo || !params_.output)
        {
            LOG_ERROR("[FusedAttentionWoStage] Null tensor pointers");
            return false;
        }

        if (params_.seq_len <= 0 || effective_kv_len <= 0 ||
            params_.n_heads <= 0 || params_.n_kv_heads <= 0 ||
            params_.head_dim <= 0 || params_.d_model <= 0)
        {
            LOG_ERROR("[FusedAttentionWoStage] Invalid dimensions");
            return false;
        }

        if (params_.n_heads % params_.n_kv_heads != 0)
        {
            LOG_ERROR("[FusedAttentionWoStage] n_heads (" << params_.n_heads
                                                          << ") must be divisible by n_kv_heads (" << params_.n_kv_heads << ")");
            return false;
        }

        // Compute position offset for decode mode
        int position_offset = params_.position_offset;
        if (position_offset == 0 && params_.seq_len < effective_kv_len)
        {
            // Auto-compute for decode mode: query is at end of cached context
            position_offset = effective_kv_len - params_.seq_len;
        }

        // Dispatch to fused kernel
        bool success = kernel_->compute(
            params_.Q,
            params_.K,
            params_.V,
            params_.Wo,
            params_.output,
            params_.seq_len,
            effective_kv_len,
            params_.causal,
            position_offset);

        if (!success)
        {
            LOG_ERROR("[FusedAttentionWoStage] Kernel compute() failed");
            return false;
        }

        LOG_DEBUG("[FusedAttentionWoStage] Execute complete");
        return true;
    }

    size_t FusedAttentionWoStage::estimatedFlops() const
    {
        // Attention FLOPs + Wo projection FLOPs
        // Attention: 2 * batch * n_heads * seq_len * kv_len * head_dim (QK^T)
        //          + 4 * batch * n_heads * seq_len * kv_len (softmax)
        //          + 2 * batch * n_heads * seq_len * kv_len * head_dim (scores @ V)
        // Wo: 2 * batch * seq_len * d_model * (n_heads * head_dim)
        const size_t qk_flops = 2ULL * params_.batch_size * params_.n_heads *
                                params_.seq_len * params_.kv_len * params_.head_dim;
        const size_t softmax_flops = 4ULL * params_.batch_size * params_.n_heads *
                                     params_.seq_len * params_.kv_len;
        const size_t sv_flops = qk_flops;
        const size_t wo_flops = 2ULL * params_.batch_size * params_.seq_len *
                                params_.d_model * (params_.n_heads * params_.head_dim);
        return qk_flops + softmax_flops + sv_flops + wo_flops;
    }

    size_t FusedAttentionWoStage::estimatedMemoryBytes() const
    {
        // Fused kernel eliminates attention context intermediate
        // Main memory: Q, K, V, Wo weights, output
        // Scratch: internal scores/state (managed by kernel)
        return static_cast<size_t>(params_.n_heads) * params_.seq_len * params_.kv_len * sizeof(float);
    }

    bool FusedAttentionWoStage::supportsBackend(ComputeBackendType backend) const
    {
        // Currently CPU-only (JIT backend uses AVX-512 VNNI)
        return backend == ComputeBackendType::CPU;
    }

    StageDumpInfo FusedAttentionWoStage::getDumpInfo() const
    {
        StageDumpInfo info;

        // Output: fused attention+Wo projection result
        if (params_.output)
        {
            const float *out_data = getSafeFp32Data(params_.output);
            if (out_data)
            {
                info.addOutput("output",
                               out_data,
                               params_.seq_len,
                               params_.d_model);
            }
        }

        info.addScalarInt("batch_size", params_.batch_size);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("kv_len", params_.kv_len);
        info.addScalarInt("n_heads", params_.n_heads);
        info.addScalarInt("n_kv_heads", params_.n_kv_heads);
        info.addScalarInt("head_dim", params_.head_dim);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarBool("causal", params_.causal);
        info.addScalarInt("backend", static_cast<int>(params_.backend));
        info.addScalarInt("device_idx", params_.device_idx);

        return info;
    }

    StageBufferRequirements FusedAttentionWoStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        // Input: Q, K, V (Q8_1)
        if (params_.Q)
        {
            const size_t q_rows = static_cast<size_t>(params_.batch_size * params_.seq_len);
            const size_t q_cols = static_cast<size_t>(params_.n_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.Q->native_type());
            reqs.addInput("Q", {q_rows, q_cols}, buf_type);
        }

        if (params_.K)
        {
            const size_t k_rows = static_cast<size_t>(params_.batch_size * params_.kv_len);
            const size_t k_cols = static_cast<size_t>(params_.n_kv_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.K->native_type());
            reqs.addInput("K", {k_rows, k_cols}, buf_type);
        }

        if (params_.V)
        {
            const size_t v_rows = static_cast<size_t>(params_.batch_size * params_.kv_len);
            const size_t v_cols = static_cast<size_t>(params_.n_kv_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.V->native_type());
            reqs.addInput("V", {v_rows, v_cols}, buf_type);
        }

        // Input: Wo weight
        if (params_.Wo)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.Wo->native_type());
            reqs.addInput("Wo", params_.Wo->shape(), buf_type);
        }

        // Output: projected attention (FP32)
        if (params_.output)
        {
            const size_t out_rows = static_cast<size_t>(params_.batch_size * params_.seq_len);
            const size_t out_cols = static_cast<size_t>(params_.d_model);
            BufferTensorType buf_type = toBufferTensorType(params_.output->native_type());
            reqs.addOutput("output", {out_rows, out_cols}, buf_type);
        }

        return reqs;
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

    StageBufferRequirements EmbeddingStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (!params_.embed_table || !params_.output)
            return reqs; // Empty if tensors not set

        // WEIGHT buffer (embedding table - read-only)
        BufferTensorType embed_type = toBufferTensorType(params_.embed_table->native_type());
        reqs.addWeight("embed_table",
                       {static_cast<size_t>(params_.vocab_size), static_cast<size_t>(params_.d_model)},
                       embed_type);

        // OUTPUT buffer (embeddings)
        BufferTensorType out_type = toBufferTensorType(params_.output->native_type());
        reqs.addOutput("output",
                       {static_cast<size_t>(params_.num_tokens), static_cast<size_t>(params_.d_model)},
                       out_type);

        // Note: token_ids is a raw int* pointer, not a TensorBase, so we don't
        // declare it as a buffer (it's typically a small CPU array)

        return reqs;
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

        // Debug: dump hidden states input at last position
        {
            const float *hidden = params_.hidden_states->fp32_data();
            if (hidden && params_.seq_len > 0)
            {
                size_t last_row_offset = (params_.seq_len - 1) * params_.d_model;
                LOG_DEBUG("[LMHeadStage] Hidden states input (last row, first 8): "
                          << hidden[last_row_offset + 0] << ","
                          << hidden[last_row_offset + 1] << ","
                          << hidden[last_row_offset + 2] << ","
                          << hidden[last_row_offset + 3] << ","
                          << hidden[last_row_offset + 4] << ","
                          << hidden[last_row_offset + 5] << ","
                          << hidden[last_row_offset + 6] << ","
                          << hidden[last_row_offset + 7]);
            }
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

        // Debug: dump local logits at last position
        {
            const float *logits_data = params_.logits->fp32_data();
            if (logits_data && params_.seq_len > 0)
            {
                // Get last row logits
                size_t last_row_offset = (params_.seq_len - 1) * params_.vocab_size;
                LOG_DEBUG("[LMHeadStage] Local logits (last row, first 8): "
                          << logits_data[last_row_offset + 0] << ","
                          << logits_data[last_row_offset + 1] << ","
                          << logits_data[last_row_offset + 2] << ","
                          << logits_data[last_row_offset + 3] << ","
                          << logits_data[last_row_offset + 4] << ","
                          << logits_data[last_row_offset + 5] << ","
                          << logits_data[last_row_offset + 6] << ","
                          << logits_data[last_row_offset + 7]);
            }
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

    StageBufferRequirements LMHeadStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (!params_.hidden_states || !params_.lm_head_weight || !params_.logits)
            return reqs; // Empty if tensors not set

        // INPUT buffer (hidden states)
        BufferTensorType hidden_type = toBufferTensorType(params_.hidden_states->native_type());
        reqs.addInput("hidden_states",
                      {static_cast<size_t>(params_.seq_len), static_cast<size_t>(params_.d_model)},
                      hidden_type);

        // WEIGHT buffer (LM head weights - may be quantized)
        BufferTensorType weight_type = toBufferTensorType(params_.lm_head_weight->native_type());
        reqs.addWeight("lm_head_weight",
                       {static_cast<size_t>(params_.vocab_size), static_cast<size_t>(params_.d_model)},
                       weight_type);

        // OUTPUT buffer (logits)
        BufferTensorType logits_type = toBufferTensorType(params_.logits->native_type());
        reqs.addOutput("logits",
                       {static_cast<size_t>(params_.seq_len), static_cast<size_t>(params_.vocab_size)},
                       logits_type);

        // Optional bias
        if (params_.bias)
        {
            reqs.addWeight("bias", {static_cast<size_t>(params_.vocab_size)}, BufferTensorType::FP32);
        }

        return reqs;
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

    std::unique_ptr<IComputeStage> ComputeStageFactory::createAllGather(
        const AllGatherStage::Params &params)
    {
        // AllGather is backend-agnostic (uses MPI directly)
        return std::make_unique<AllGatherStage>(params);
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

    std::unique_ptr<IComputeStage> ComputeStageFactory::createKVCacheGather(
        const KVCacheGatherStage::Params &params)
    {
        // KV cache gather is backend-agnostic (pure memory operations)
        return std::make_unique<KVCacheGatherStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createAttentionCompute(
        const AttentionComputeStage::Params &params)
    {
        // Unified: AttentionComputeStage uses KernelFactory at execute-time
        return std::make_unique<AttentionComputeStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createFusedAttentionWo(
        const FusedAttentionWoStage::Params &params)
    {
        // Fused attention + Wo projection using JIT kernel
        return std::make_unique<FusedAttentionWoStage>(params);
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
