/**
 * @file StageVerifier.cpp
 * @brief Stage input/output verification implementation
 * @author David Sanftenberg
 * @date December 2025
 *
 * Original input/output declarations retain tensor/subview and publication
 * ownership. Authored request intervals validate only numerical rows produced by
 * the current transaction; corrupt native metadata never becomes a valid empty
 * diagnostic. GPU counts stay native until the existing compact result retires.
 * Extracted from DeviceGraphExecutor.cpp.
 */

#include "StageVerifier.h"

#if LLAMINAR_ASSERTIONS_ACTIVE

#include "../../../tensors/TensorVerification.h"
#include "../../../tensors/GPUTensorVerification.h"
#include "../../../snapshots/SnapshotSemanticObservation.h"
#include <limits>
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
            bool passed = !gpu_result.invalid_extent;
            if (!passed)
                reason << "invalid acquired output extent";

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
            if (config.check_all_zero && gpu_result.total_checked != 0 && gpu_result.appears_zero)
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
         * @brief Bind an input or output's exact authored axis to its original publication.
         * @param buffer Original dump declaration, including input query elision intervals.
         * @param gpu Whether the payload's authoritative storage is native GPU memory.
         * @param device Exact stage/metadata device, never inferred from a count value.
         * @return Call-only borrow, valid until this validation transaction retires.
         * @throws std::logic_error for missing/foreign/malformed frontier ownership.
         */
        template <typename Buffer>
        [[nodiscard]] TensorValidationExtent validationExtent(
            const Buffer &buffer, bool gpu, DeviceId device)
        {
            TensorValidationExtent result{
                {SnapshotContiguousRowKind::Physical, buffer.rows}, buffer.cols, nullptr};
            if constexpr (requires { buffer.semantic_extent; })
            {
                if(const auto *authored=std::get_if<SnapshotAuthoredTokenRows>(&buffer.semantic_extent)) {
                    result.projectionWork=snapshotProjectionWork(*authored,gpu,device);
                    result.rowSelection=TensorValidationRowSelection::ProjectionWork;return result;
                }
                SnapshotInt32Binding binding;
                std::size_t words = 1;
                if (const auto *tokens = std::get_if<SnapshotTokenRows>(&buffer.semantic_extent))
                {
                    result.geometry.kind = SnapshotContiguousRowKind::TokenRows;
                    binding = tokens->liveRows;
                }
                else if (const auto *state = std::get_if<SnapshotStatePrefixRows>(&buffer.semantic_extent))
                {
                    result.geometry.kind = SnapshotContiguousRowKind::StatePrefixRows;
                    result.geometry.sourceCapacity = state->sourceCapacity;
                    result.geometry.sourceRowsPerStateRow = state->sourceRowsPerStateRow;
                    binding = state->frontier;
                    words = 2;
                    if (!binding.tensor)
                        throw std::logic_error("State-prefix validation requires its original frontier owner");
                }
                else if (!std::holds_alternative<std::monostate>(buffer.semantic_extent))
                    throw std::logic_error("Floating validation requires a contiguous authored output axis");
                if (gpu && binding.tensor && !binding.tensor->isDeviceValid())
                    throw std::logic_error("Output frontier has no authoritative native storage");
                if (!gpu && binding.tensor && binding.tensor->isDeviceValid() &&
                    binding.tensor->current_device() && binding.tensor->current_device()->is_gpu())
                    throw std::logic_error("CPU validation cannot consume a GPU-owned frontier");
                result.frontier = snapshotCountPointer(binding, words, gpu, device);
            }
            return result;
        }

        /**
         * @brief Validate only authored output rows without changing residency authority.
         *
         * Static bounds and original owners are checked even when the acquired
         * extent is empty. GPU counts are read only by the existing initializer
         * on the exact stage stream; the sole compact result distinguishes an
         * authentic empty observation from malformed native metadata. Integer
         * formats keep their existing no-NaN/Inf validation policy.
         */
        template <typename Buffer>
        [[nodiscard]] BufferValidation validateBuffer(
            const Buffer &buffer,
            IComputeStage &stage,
            const verification::VerificationConfig &config)
        {
            bool gpu = false;
            try
            {
                if (buffer.cols && buffer.rows > std::numeric_limits<size_t>::max() / buffer.cols)
                    throw std::logic_error("Stage buffer element extent overflows");
                const size_t numel = buffer.rows * buffer.cols;
                if (!numel)
                {
                    if constexpr (requires { buffer.semantic_extent; })
                    {
                        if (!std::holds_alternative<std::monostate>(buffer.semantic_extent))
                            throw std::logic_error("Authored output omits its positive physical capacity");
                    }
                    return {verification::VerificationResult::ok(), false};
                }
                ITensor *tensor = buffer.tensor;
                gpu = tensor && tensor->isDeviceValid();
                const auto device = gpu ? tensor->current_device() : std::optional<DeviceId>{};
                if (gpu && (!device || !device->is_gpu() || *device != stage.device()))
                    throw std::logic_error("Authoritative tensor device does not match its stage device");
                const auto data_type = gpuValidationDataType(buffer.dtype);
                if (!data_type)
                    return {verification::VerificationResult::ok(), gpu};
                if (!buffer.element_size || numel > std::numeric_limits<size_t>::max() / buffer.element_size)
                    throw std::logic_error("Stage buffer byte extent overflows");
                const auto bytes = numel * buffer.element_size;
                const auto expected_width = *data_type == TensorValidationDataType::FP32 ? 4U : 2U;
                if (buffer.element_size != expected_width || buffer.tensor_byte_offset % expected_width ||
                    (buffer.byte_size && bytes > buffer.byte_size))
                    throw std::logic_error("Stage floating subview has inconsistent scalar/byte geometry");
                if (tensor && (buffer.tensor_byte_offset > tensor->size_bytes() ||
                    bytes > tensor->size_bytes() - buffer.tensor_byte_offset))
                    throw std::logic_error("Stage floating subview exceeds its original tensor owner");
                if (gpu)
                {
                    const auto *base = static_cast<const std::byte *>(tensor->gpu_data_ptr());
                    if (!base)
                        throw std::logic_error("Device-valid tensor exposes a null device pointer");
                    const auto extent = validationExtent(buffer, true, *device);
                    auto *validator = getTensorValidator(device->type, device->ordinal);
                    if (!validator)
                        throw std::logic_error("No device validator is installed for " + device->to_string());
                    const auto result = validator->validateObserved(
                        base + buffer.tensor_byte_offset, extent, *data_type,
                        ExplicitGPUStream{stage.gpuStream()});
                    return {gpuResultToVerification(buffer.name, result, config), true};
                }
                const auto extent = validationExtent(buffer, false, stage.device());
                const bool authored = extent.usesProjectionWork() || extent.geometry.kind != SnapshotContiguousRowKind::Physical;
                const void *data = buffer.data;
                if (authored)
                {
                    if (!tensor || !tensor->isHostValid() || stage.device().is_gpu())
                        throw std::logic_error("Authored CPU output requires its original host-owned payload");
                    const auto *base = static_cast<const std::byte *>(tensor->raw_data());
                    if (!base)
                        throw std::logic_error("Authored CPU output exposes a null host pointer");
                    data = base + buffer.tensor_byte_offset;
                }
                if (!data)
                    throw std::logic_error("Stage buffer exposes a null payload pointer");
                if(extent.usesProjectionWork()) {
                    auto total=verification::VerificationResult::ok();
                    total.tensor_name=bufferName(buffer.name);
                    auto perRequest=config;perRequest.check_all_zero=false;
                    for(int request=extent.projectionWork.requestForRow(0);request<=extent.projectionWork.requestForRow(extent.projectionWork.physicalRows()-1);++request) {
                        const auto span=extent.projectionWork.resolveHost(request);
                        if(!span.size())continue;
                        // Retain original physical coordinates; never scan poison
                        // in a short request's prefix or inactive bucket suffix.
                        const auto result=verification::verifyRawBuffer(
                            static_cast<const std::byte *>(data)+static_cast<std::size_t>(span.first)*buffer.cols*buffer.element_size,
                            static_cast<std::size_t>(span.size()),buffer.cols,bufferName(buffer.name),buffer.dtype,perRequest);
                        if(!result.passed) return {result,false};
                        total.nan_count+=result.nan_count;total.inf_count+=result.inf_count;
                        total.zero_count+=result.zero_count;total.total_sampled+=result.total_sampled;
                    }
                    if(config.check_all_zero&&total.total_sampled&&total.zero_count==total.total_sampled)
                        return {verification::VerificationResult::withDiagnostics(false,bufferName(buffer.name),
                            "All authored sampled elements are zero",total.nan_count,total.inf_count,total.zero_count,total.total_sampled),false};
                    return {total,false};
                }
                std::int32_t prior = 0, live = 0;
                auto geometry = extent.geometry;
                if (geometry.kind == SnapshotContiguousRowKind::StatePrefixRows)
                {
                    prior = extent.frontier[0];
                    live = extent.frontier[1];
                }
                else if (geometry.kind == SnapshotContiguousRowKind::TokenRows && extent.frontier)
                    live = extent.frontier[0];
                else
                    geometry.kind = SnapshotContiguousRowKind::Physical;
                const auto observed = resolveSnapshotContiguousRows(geometry, prior, live);
                if (!observed.valid)
                    throw std::logic_error("Invalid acquired output extent");
                return {verification::verifyRawBuffer(data, observed.rows, buffer.cols,
                    bufferName(buffer.name), buffer.dtype, config), false};
            }
            catch (const std::exception &error)
            {
                return {verification::VerificationResult::fail(bufferName(buffer.name),
                    std::string("Buffer validation failed fatally: ") + error.what()), gpu};
            }
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
