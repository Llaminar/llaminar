/**
 * @file StageVerifier.cpp
 * @brief Stage input/output verification implementation
 * @author David Sanftenberg
 * @date December 2025
 *
 * Extracted from DeviceGraphExecutor.cpp.
 */

#include "StageVerifier.h"

#if LLAMINAR_ASSERTIONS_ACTIVE

#include "../../../tensors/TensorVerification.h"
#include "../../../tensors/GPUTensorVerification.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../utils/Logger.h"
#include "../../../utils/DebugEnv.h"
#include <cmath>
#include <string>

namespace llaminar2
{
    namespace
    {
        bool fp32BufferIsAllZero(const float *data, size_t numel)
        {
            if (!data || numel == 0)
                return false;

            for (size_t i = 0; i < numel; ++i)
            {
                if (data[i] != 0.0f)
                    return false;
            }

            return true;
        }
    } // namespace

    void verifyStageEntry(const ComputeNode &node, int layer_idx)
    {
        using namespace verification;

        const auto &validation = debugEnv().validation;
        auto dump_info = node.stage->getDumpInfo();

        // Build verification config from global settings
        VerificationConfig vconfig;
        vconfig.sample_rows = validation.sample_rows;
        vconfig.check_null = true;
        vconfig.check_nan = validation.fail_on_nan;
        vconfig.check_inf = validation.fail_on_nan; // Inf is also bad
        vconfig.check_all_zero = false;             // Zero inputs may be valid (first layer residual)
        vconfig.dump_on_failure = validation.dump_on_failure;

        // Verify all inputs (NaN/Inf/null checks)
        for (const auto &input : dump_info.inputs)
        {
            if (!input.data)
                continue; // Null inputs checked separately if needed

            auto result = verifyRawBuffer(
                input.data, input.rows, input.cols,
                input.name, input.dtype, vconfig);

            if (!result.passed)
            {
                LOG_ERROR("[VERIFY] ENTRY FAILED: layer=" << layer_idx
                                                          << " stage=" << node.name
                                                          << " tensor=" << result.tensor_name
                                                          << " reason=" << result.error_reason);

                // Dump all buffers for debugging
                std::string dump_path;
                if (vconfig.dump_on_failure)
                {
                    dump_path = dumpStageBuffers(node.name, layer_idx, "ENTRY", dump_info,
                                                 result.tensor_name, result.error_reason);
                    LOG_ERROR("[VERIFY] Buffers dumped to: " << dump_path);
                }

                // Throw exception with full context
                throw VerificationFailure(node.name, layer_idx, "ENTRY",
                                          result.tensor_name, result.error_reason, dump_path);
            }
        }

        // =====================================================================
        // Phase 3: Automatic Layout Validation (declarative)
        // If stage provides LayoutExpectation, validate all buffers with layouts
        // =====================================================================
        if (validation.validate_inputs)
        {
            auto layout_expect = node.stage->getLayoutExpectation();
            if (layout_expect.is_set())
            {
                auto buf_reqs = node.stage->getBufferRequirements();
                for (const auto &buf : buf_reqs.buffers)
                {
                    // Only validate buffers with declared layouts
                    if (buf.expected_layout == TensorLayout::UNKNOWN)
                        continue;

                    // Only validate INPUT buffers at entry
                    if (buf.role != BufferRole::INPUT && buf.role != BufferRole::INOUT)
                        continue;

                    auto result = validateBufferLayoutByShape(
                        buf.shape, buf.name.c_str(),
                        buf.expected_layout, layout_expect);

                    if (!result.passed)
                    {
                        LOG_ERROR("[VERIFY] LAYOUT FAILED: layer=" << layer_idx
                                                                   << " stage=" << node.name
                                                                   << " buffer=" << buf.name
                                                                   << " reason=" << result.error_reason);

                        std::string dump_path;
                        if (vconfig.dump_on_failure)
                        {
                            dump_path = dumpStageBuffers(node.name, layer_idx, "ENTRY_LAYOUT", dump_info,
                                                         buf.name, result.error_reason);
                        }

                        throw VerificationFailure(node.name, layer_idx, "ENTRY_LAYOUT",
                                                  buf.name, result.error_reason, dump_path);
                    }
                }
            }
        }
    }

    void verifyStageExit(const ComputeNode &node, int layer_idx)
    {
        using namespace verification;

        const auto &validation = debugEnv().validation;
        auto dump_info = node.stage->getDumpInfo();

        // Build verification config from global settings
        VerificationConfig vconfig;
        vconfig.sample_rows = validation.sample_rows;
        vconfig.check_null = true;
        vconfig.check_nan = validation.fail_on_nan;
        vconfig.check_inf = validation.fail_on_nan;
        vconfig.dump_on_failure = validation.dump_on_failure;

        // All-zero output check: enabled by config UNLESS stage explicitly allows zero outputs
        // Most stages should never produce all-zero outputs (indicates bugs).
        // Stages like KVCacheGatherStage can override allowsZeroOutput() to return true.
        vconfig.check_all_zero = validation.fail_on_zero && !node.stage->allowsZeroOutput();

        // Verify all outputs (NaN/Inf/null checks)
        // IMPORTANT: Sync outputs from GPU BEFORE reading data
        dump_info.ensureOutputsOnHost();

        for (const auto &output : dump_info.outputs)
        {
            if (!output.data)
                continue;

            auto result = verifyRawBuffer(
                output.data, output.rows, output.cols,
                output.name, output.dtype, vconfig);

            if (!result.passed)
            {
                LOG_ERROR("[VERIFY] EXIT FAILED: layer=" << layer_idx
                                                         << " stage=" << node.name
                                                         << " tensor=" << result.tensor_name
                                                         << " reason=" << result.error_reason);

                // Dump all buffers for debugging
                std::string dump_path;
                if (vconfig.dump_on_failure)
                {
                    dump_path = dumpStageBuffers(node.name, layer_idx, "EXIT", dump_info,
                                                 result.tensor_name, result.error_reason);
                    LOG_ERROR("[VERIFY] Buffers dumped to: " << dump_path);
                }

                // Throw exception with full context
                throw VerificationFailure(node.name, layer_idx, "EXIT",
                                          result.tensor_name, result.error_reason, dump_path);
            }
        }

        // =====================================================================
        // Phase 3: Automatic Layout Validation (declarative)
        // Validate OUTPUT buffers with declared layouts at stage exit
        // =====================================================================
        if (validation.validate_buffers)
        {
            auto layout_expect = node.stage->getLayoutExpectation();
            if (layout_expect.is_set())
            {
                auto buf_reqs = node.stage->getBufferRequirements();
                for (const auto &buf : buf_reqs.buffers)
                {
                    // Only validate buffers with declared layouts
                    if (buf.expected_layout == TensorLayout::UNKNOWN)
                        continue;

                    // Only validate OUTPUT buffers at exit
                    if (buf.role != BufferRole::OUTPUT && buf.role != BufferRole::INOUT)
                        continue;

                    auto result = validateBufferLayoutByShape(
                        buf.shape, buf.name.c_str(),
                        buf.expected_layout, layout_expect);

                    if (!result.passed)
                    {
                        LOG_ERROR("[VERIFY] LAYOUT FAILED: layer=" << layer_idx
                                                                   << " stage=" << node.name
                                                                   << " buffer=" << buf.name
                                                                   << " reason=" << result.error_reason);

                        std::string dump_path;
                        if (vconfig.dump_on_failure)
                        {
                            dump_path = dumpStageBuffers(node.name, layer_idx, "EXIT_LAYOUT", dump_info,
                                                         buf.name, result.error_reason);
                        }

                        throw VerificationFailure(node.name, layer_idx, "EXIT_LAYOUT",
                                                  buf.name, result.error_reason, dump_path);
                    }
                }
            }
        }
    }

    bool validateStageOutputs(const ComputeNode &node)
    {
        if (!node.stage)
            return true;

        const auto &validation = debugEnv().validation;

        // Get stage's dump info to access output buffers
        // NOTE: We intentionally access getDumpInfo() here even though it may trigger
        // GPU→host sync, because we only call this in Debug/Integration builds anyway.
        // The StageDumpInfo provides tensor pointers that we can use for GPU validation.
        auto dump_info = node.stage->getDumpInfo();
        const bool zero_output_allowed = node.stage->allowsZeroOutput();

        bool all_valid = true;

        // Validate output buffers
        for (const auto &output : dump_info.outputs)
        {
            if (output.rows == 0 || output.cols == 0)
                continue;

            size_t numel = output.rows * output.cols;

            // Check if we have a tensor pointer with GPU data
            // If so, use GPU-side validation to avoid expensive D2H sync
            if (output.tensor)
            {
                auto *base_tensor = dynamic_cast<TensorBase *>(output.tensor);
                if (base_tensor && base_tensor->deviceValid())
                {
                    // Get GPU validator for this device type
                    auto device_opt = base_tensor->current_device();
                    if (device_opt.has_value())
                    {
                        ITensorValidator *validator = getTensorValidator(device_opt->type);
                        if (validator)
                        {
                            const void *device_ptr = base_tensor->gpu_data_ptr();
                            int device_id = device_opt->ordinal;

                            // Launch GPU validation kernel (async)
                            bool launched = false;
                            if (std::string(output.dtype) == "FP32")
                            {
                                launched = validator->validateFP32Async(device_ptr, numel, device_id);
                            }
                            else if (std::string(output.dtype) == "BF16")
                            {
                                launched = validator->validateBF16Async(device_ptr, numel, device_id);
                            }
                            else if (std::string(output.dtype) == "FP16")
                            {
                                launched = validator->validateFP16Async(device_ptr, numel, device_id);
                            }

                            if (launched)
                            {
                                TensorValidationResult result;
                                if (validator->getResult(result))
                                {
                                    if (result.appears_zero && numel > 10 && !zero_output_allowed)
                                    {
                                        LOG_WARN("[StageVerifier] Stage '" << node.name << "' output '" << output.name
                                                                           << "' appears to be all zeros (GPU validation)");
                                        if (validation.fail_on_zero)
                                        {
                                            LOG_ERROR("[StageVerifier] Buffer validation failed: zero tensor detected");
                                            all_valid = false;
                                        }
                                    }

                                    if (result.has_nan || result.has_inf)
                                    {
                                        LOG_WARN("[StageVerifier] Stage '" << node.name << "' output '" << output.name
                                                                           << "' contains " << result.nan_count << " NaN, "
                                                                           << result.inf_count << " Inf values (GPU validation)");
                                        if (validation.fail_on_nan)
                                        {
                                            LOG_ERROR("[StageVerifier] Buffer validation failed: NaN/Inf detected");
                                            all_valid = false;
                                        }
                                    }

                                    // Successfully validated on GPU, continue to next output
                                    continue;
                                }
                            }
                            // Fall through to host validation if GPU validation failed to launch
                        }
                    }
                }
            }

            // Fallback: Host-side validation (for CPU tensors or when GPU validation unavailable)
            // Need to ensure output is synced to host before reading
            if (output.tensor)
            {
                if (auto *cpu_tensor = dynamic_cast<TensorBase *>(output.tensor))
                {
                    cpu_tensor->ensureOnHost();
                }
            }
            if (!output.data)
                continue;

            if (std::string(output.dtype) == "FP32")
            {
                const float *fp32_data = static_cast<const float *>(output.data);

                // Quick zero check: sample first, middle, last elements
                bool appears_zero = true;
                if (numel > 0 && fp32_data[0] != 0.0f)
                    appears_zero = false;
                if (numel > 1 && fp32_data[numel / 2] != 0.0f)
                    appears_zero = false;
                if (numel > 2 && fp32_data[numel - 1] != 0.0f)
                    appears_zero = false;

                // Full check if samples all zero
                if (appears_zero && numel > 3)
                {
                    size_t sample_stride = std::max(size_t(1), numel / 100);
                    for (size_t i = 0; i < numel; i += sample_stride)
                    {
                        if (fp32_data[i] != 0.0f)
                        {
                            appears_zero = false;
                            break;
                        }
                    }
                }

                if (appears_zero)
                    appears_zero = fp32BufferIsAllZero(fp32_data, numel);

                if (appears_zero && !zero_output_allowed)
                {
                    LOG_WARN("[StageVerifier] Stage '" << node.name << "' output '" << output.name
                                                       << "' appears to be all zeros (likely uninitialized)");

                    if (validation.fail_on_zero)
                    {
                        LOG_ERROR("[StageVerifier] Buffer validation failed: zero tensor detected");
                        all_valid = false;
                    }
                }

                // Check for NaN/Inf
                bool has_nan_inf = false;
                for (size_t i = 0; i < numel && !has_nan_inf; i += std::max(size_t(1), numel / 100))
                {
                    if (std::isnan(fp32_data[i]) || std::isinf(fp32_data[i]))
                    {
                        has_nan_inf = true;
                    }
                }

                if (has_nan_inf)
                {
                    LOG_WARN("[StageVerifier] Stage '" << node.name << "' output '" << output.name
                                                       << "' contains NaN or Inf values");

                    if (validation.fail_on_nan)
                    {
                        LOG_ERROR("[StageVerifier] Buffer validation failed: NaN/Inf detected");
                        all_valid = false;
                    }
                }
            }
        }

        return all_valid;
    }

} // namespace llaminar2

#endif // LLAMINAR_ASSERTIONS_ACTIVE
