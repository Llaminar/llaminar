/**
 * @file ComputeStageBase.cpp
 * @brief Implementation of ComputeStageBase
 */

#include "IComputeStage.h"
#include "ComputeStageUtils.h"
#include "../../utils/DebugEnv.h"
#include "../../tensors/Tensors.h"
#include "../../tensors/TensorVerification.h"
#include "../../utils/Logger.h"
#include "../../kernels/KernelFactory.h"
#include <chrono>

namespace llaminar2
{

    // =============================================================================
    // Default Layout Expectation (Phase 3: Tensor Layout Contracts)
    // =============================================================================

    LayoutExpectation IComputeStage::getLayoutExpectation() const
    {
        // Default: return empty expectation (no automatic layout validation)
        // Stages override this to enable declarative layout validation
        return LayoutExpectation{};
    }

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
        case ComputeStageType::ALLGATHER_V:
            return "ALLGATHER_V";
        case ComputeStageType::SEND_ACTIVATIONS:
            return "SEND_ACTIVATIONS";
        case ComputeStageType::RECV_ACTIVATIONS:
            return "RECV_ACTIVATIONS";
        case ComputeStageType::LOCAL_PP_TRANSFER:
            return "LOCAL_PP_TRANSFER";
        case ComputeStageType::GLOBAL_PP_TRANSFER:
            return "GLOBAL_PP_TRANSFER";
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
        case ComputeStageType::QK_NORM:
            return "QK_NORM";
        default:
            return "UNKNOWN";
        }
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

    void IComputeStage::traceInput(const std::string &tensor_name, const ITensor *tensor) const
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

    void IComputeStage::traceOutput(const std::string &tensor_name, const ITensor *tensor) const
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

    void IComputeStage::validateShape(const std::string &tensor_name, const ITensor *tensor,
                                      const std::vector<size_t> &expected) const
    {
        validateShapes({{tensor_name, tensor, expected, false}});
    }

    void IComputeStage::validateMatmulShapes(const std::string &a_name, const ITensor *a,
                                             const std::string &b_name, const ITensor *b) const
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

    StageDumpInfo &StageDumpInfo::addWeight(const char *name, const ITensor *tensor)
    {
        if (tensor)
        {
            weights.push_back({name, tensor, nullptr, 0,
                               tensor->rows(), tensor->cols(),
                               tensor->dtype_name()});
        }
        return *this;
    }

    StageDumpInfo &StageDumpInfo::addInput(const char *name, const ITensor *tensor, size_t rows, size_t cols)
    {
        // Store raw tensor data in native format for accurate replay testing
        // This preserves quantized blocks (Q8_1, Q16_1, etc.) instead of dequantizing
        //
        // CRITICAL: Use the LOGICAL dimensions (rows, cols) to compute byte_size,
        // NOT tensor->size_bytes() which may reflect a larger pre-allocated buffer
        // (e.g., KV cache allocated for max_seq_len but only using kv_len positions)
        const void *data = tensor ? tensor->raw_data() : nullptr;
        const char *dtype = tensor ? tensor->dtype_name() : "FP32";

        LOG_DEBUG("[StageDumpInfo::addInput] name=" << name
                                                    << " tensor=" << (tensor ? "non-null" : "null")
                                                    << " native_type=" << (tensor ? static_cast<int>(tensor->native_type()) : -1)
                                                    << " dtype_name=" << dtype);

        // Special handling for Q16_1: use block-size-aware dtype name
        // Q16_1Tensor has variable block sizes (32, 64, 128 elements)
        if (tensor && tensor->native_type() == TensorType::Q16_1)
        {
            const auto *q16_tensor = dynamic_cast<const Q16_1Tensor *>(tensor);
            if (q16_tensor)
            {
                dtype = q16_tensor->dtype_name_with_block_size();
                LOG_DEBUG("[StageDumpInfo::addInput] Q16_1 detected, block-size dtype=" << dtype);
            }
            else
            {
                LOG_WARN("[StageDumpInfo::addInput] Q16_1 native_type but dynamic_cast failed!");
            }
        }

        // Compute byte size from logical dimensions and dtype
        size_t byte_size = computeByteSizeForDtype(dtype, rows, cols);
        LOG_DEBUG("[StageDumpInfo::addInput] Final dtype=" << dtype << " byte_size=" << byte_size);
        size_t element_size = (rows > 0 && cols > 0) ? byte_size / (rows * cols) : sizeof(float);

        InputBuffer buf{name, data, rows, cols, dtype, element_size, byte_size};
        buf.tensor = const_cast<ITensor *>(tensor); // Store tensor pointer for coherence
        inputs.push_back(buf);
        return *this;
    }

    StageDumpInfo &StageDumpInfo::addOutput(const char *name, const ITensor *tensor, size_t rows, size_t cols)
    {
        // Store raw tensor data in native format for accurate replay testing
        // This preserves quantized blocks (Q8_1, Q16_1, etc.) instead of dequantizing
        //
        // CRITICAL: Use the LOGICAL dimensions (rows, cols) to compute byte_size,
        // NOT tensor->size_bytes() which may reflect a larger pre-allocated buffer

        // NOTE: We NO LONGER call ensureOnHost() here!
        // The sync is now DEFERRED to ensureOutputsOnHost() which is called
        // only when data is actually needed (e.g., verification, dumping).
        // This allows GPU kernels to run async without blocking.

        const void *data = tensor ? tensor->raw_data() : nullptr;
        const char *dtype = tensor ? tensor->dtype_name() : "FP32";

        // Special handling for Q16_1: use block-size-aware dtype name
        // Q16_1Tensor has variable block sizes (32, 64, 128 elements)
        if (tensor && tensor->native_type() == TensorType::Q16_1)
        {
            const auto *q16_tensor = dynamic_cast<const Q16_1Tensor *>(tensor);
            if (q16_tensor)
            {
                dtype = q16_tensor->dtype_name_with_block_size();
            }
        }

        // Compute byte size from logical dimensions and dtype
        size_t byte_size = computeByteSizeForDtype(dtype, rows, cols);
        size_t element_size = (rows > 0 && cols > 0) ? byte_size / (rows * cols) : sizeof(float);

        OutputBuffer buf{name, data, rows, cols, dtype, element_size, byte_size};
        buf.tensor = const_cast<ITensor *>(tensor); // Store tensor pointer for coherence
        outputs.push_back(buf);
        return *this;
    }

    void StageDumpInfo::ensureOutputsOnHost() const
    {
        // Sync all output tensors from GPU to host.
        // Call this BEFORE reading output.data for verification/dumping.
        for (auto &output : outputs)
        {
            if (output.tensor)
            {
                if (auto *cpu_tensor = dynamic_cast<TensorBase *>(output.tensor))
                {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    cpu_tensor->ensureOnHost();
                    auto t1 = std::chrono::high_resolution_clock::now();
                    auto elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    if (elapsed_ms > 1.0)
                    {
                        LOG_TRACE("[StageDumpInfo::ensureOutputsOnHost] '" << output.name << "' took " << elapsed_ms << " ms");
                    }
                    // Update data pointer after sync (may have changed)
                    output.data = cpu_tensor->raw_data();
                }
            }
        }
    }

} // namespace llaminar2
