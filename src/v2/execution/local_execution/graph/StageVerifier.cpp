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
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

namespace llaminar2
{
    namespace
    {
        struct BufferValidation
        {
            verification::VerificationResult result;
            bool used_device_validator = false;
        };

        [[nodiscard]] const char *bufferName(const char *name) noexcept
        {
            return name ? name : "<unnamed>";
        }

        [[nodiscard]] std::optional<TensorValidationDataType>
        gpuValidationDataType(const char *dtype)
        {
            const std::string_view name = dtype ? std::string_view(dtype) : std::string_view{};
            if (name == "FP32")
                return TensorValidationDataType::FP32;
            if (name == "BF16")
                return TensorValidationDataType::BF16;
            if (name == "FP16")
                return TensorValidationDataType::FP16;
            return std::nullopt;
        }

        [[nodiscard]] verification::VerificationResult gpuResultToVerification(
            const char *name,
            const TensorValidationResult &gpu_result,
            const verification::VerificationConfig &config)
        {
            std::ostringstream reason;
            bool passed = true;

            if (config.check_nan && gpu_result.has_nan)
            {
                passed = false;
                reason << "Contains " << gpu_result.nan_count << " NaN values";
            }
            if (config.check_inf && gpu_result.has_inf)
            {
                if (!passed)
                    reason << "; ";
                passed = false;
                reason << "contains " << gpu_result.inf_count << " Inf values";
            }
            if (config.check_all_zero && gpu_result.appears_zero)
            {
                if (!passed)
                    reason << "; ";
                passed = false;
                reason << "all " << gpu_result.total_checked
                       << " device elements are zero (likely uninitialized)";
            }

            return verification::VerificationResult::withDiagnostics(
                passed,
                bufferName(name),
                reason.str(),
                gpu_result.nan_count,
                gpu_result.inf_count,
                gpu_result.zero_count,
                gpu_result.total_checked);
        }

        /**
         * Validate one stage buffer without changing its residency authority.
         *
         * GPU tensors are scanned in place on the exact stage stream. A missing
         * validator, stream, current device, or device pointer is an
         * infrastructure failure and is reported as a failed verification; the
         * caller never attempts to repair it by materializing the tensor on the
         * host. Non-floating formats have no NaN/Inf semantics and retain the
         * same explicit no-op validation contract as verifyRawBuffer().
         */
        template <typename Buffer>
        [[nodiscard]] BufferValidation validateBuffer(
            const Buffer &buffer,
            IComputeStage &stage,
            const verification::VerificationConfig &config)
        {
            const size_t numel = buffer.rows * buffer.cols;
            if (numel == 0)
                return {verification::VerificationResult::ok(), false};

            ITensor *tensor = buffer.tensor;
            if (tensor && tensor->isDeviceValid())
            {
                const auto device = tensor->current_device();
                if (!device || !device->is_gpu())
                {
                    return {verification::VerificationResult::fail(
                                bufferName(buffer.name),
                                "device-valid tensor has no owning GPU device"),
                            true};
                }

                if (*device != stage.device())
                {
                    return {verification::VerificationResult::fail(
                                bufferName(buffer.name),
                                "authoritative tensor device " + device->to_string() +
                                    " does not match stage device " + stage.device().to_string()),
                            true};
                }

                const auto data_type = gpuValidationDataType(buffer.dtype);
                if (!data_type)
                    return {verification::VerificationResult::ok(), true};

                const void *device_ptr = tensor->gpu_data_ptr();
                if (!device_ptr)
                {
                    return {verification::VerificationResult::fail(
                                bufferName(buffer.name),
                                "device-valid tensor exposes a null device pointer"),
                            true};
                }

                ITensorValidator *validator = getTensorValidator(device->type, device->ordinal);
                if (!validator)
                {
                    return {verification::VerificationResult::fail(
                                bufferName(buffer.name),
                                "no device validator is installed for " + device->to_string()),
                            true};
                }

                try
                {
                    const auto gpu_result = validator->validate(
                        device_ptr,
                        numel,
                        *data_type,
                        ExplicitGPUStream{stage.gpuStream()});
                    return {gpuResultToVerification(buffer.name, gpu_result, config), true};
                }
                catch (const std::exception &error)
                {
                    return {verification::VerificationResult::fail(
                                bufferName(buffer.name),
                                std::string("device validation failed fatally: ") + error.what()),
                            true};
                }
            }

            return {verification::verifyRawBuffer(
                        buffer.data,
                        buffer.rows,
                        buffer.cols,
                        bufferName(buffer.name),
                        buffer.dtype,
                        config),
                    false};
        }
    } // namespace

    void verifyStageEntry(const ComputeNode &node, int layer_idx)
    {
        using namespace verification;

        const auto &validation = debugEnv().validation;
        StageDumpInfo dump_info = node.stage->getDumpInfoSnapshot();

        // Build verification config from global settings
        VerificationConfig vconfig;
        vconfig.sample_rows = validation.sample_rows;
        vconfig.check_null = true;
        vconfig.check_nan = validation.fail_on_nan;
        vconfig.check_inf = validation.fail_on_nan; // Inf is also bad
        vconfig.check_all_zero = false;             // Zero inputs may be valid (first layer residual)
        vconfig.dump_on_failure = validation.dump_on_failure;

        // Verify inputs without changing residency. GPU inputs are already
        // ordered onto the consumer stage stream by the arena dependency DAG.
        for (const auto &input : dump_info.inputs)
        {
            auto validation_result = validateBuffer(input, *node.stage, vconfig);
            const auto &result = validation_result.result;

            if (!result.passed)
            {
                LOG_ERROR("[VERIFY] ENTRY FAILED: layer=" << layer_idx
                                                          << " stage=" << node.name
                                                          << " tensor=" << result.tensor_name
                                                          << " reason=" << result.error_reason);

                // Dump all buffers for debugging
                std::string dump_path;
                if (vconfig.dump_on_failure && !validation_result.used_device_validator)
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
        StageDumpInfo dump_info = node.stage->getDumpInfoSnapshot();

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

        // Verify outputs at their authoritative residency. Ordinary GPU
        // validation never downloads the tensor payload or changes coherence.
        for (const auto &output : dump_info.outputs)
        {
            auto validation_result = validateBuffer(output, *node.stage, vconfig);
            const auto &result = validation_result.result;

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
                    // Failure dumping is an explicit terminal diagnostic. It is
                    // never used as validation or as a recovery path.
                    if (validation_result.used_device_validator)
                        dump_info.ensureOutputsOnHost(node.stage->gpuStream());
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

} // namespace llaminar2

#endif // LLAMINAR_ASSERTIONS_ACTIVE
