/**
 * @file MoEExpertParallelReduceStage.cpp
 * @brief Implementation of cross-domain partial reduction for MoE expert-parallel tiers.
 *
 * Bridge Phase 7A sparse return interface:
 *  - Partials with selected_rows non-empty are scatter-added (sparse layout).
 *  - Partials with selected_rows empty use element-wise accumulation (dense layout).
 *  - GPU-native scatter-add for ContinuationDeviceOptimized is deferred; sparse
 *    partials in optimized mode use caller-provided dense expansion buffers, then
 *    H2D upload and standard device accumulation. See runContinuationDeviceOptimizedReduce().
 */

#include "MoEExpertParallelReduceStage.h"

#include "../../../backends/BackendManager.h"
#include "../../../backends/IBackend.h"
#include "../../../execution/moe/IOverlayDomainRuntime.h"
#include "../../../execution/moe/MoEExpertOverlayProfiler.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/Logger.h"

#include <algorithm>
#include <chrono>
#include <sstream>
#include <utility>
#include <vector>

namespace llaminar2
{
namespace
{
    constexpr size_t kUnknownSize = 0;

    struct PartialRuntime
    {
        const ITensor *tensor = nullptr;
        MoEExpertParallelReducePartialInfo info;
        size_t bytes = 0;
        bool host_sync_required = false;
        bool is_sparse = false;       ///< True when info.selected_rows is non-empty
        size_t sparse_row_count = 0;  ///< Populated from info.selected_rows.size() when is_sparse
    };

    bool validateFP32Tensor(const ITensor *tensor, const char *name)
    {
        if (!tensor)
        {
            LOG_ERROR("[MoEExpertParallelReduceStage] Null " << name << " tensor");
            return false;
        }
        if (tensor->native_type() != TensorType::FP32)
        {
            LOG_ERROR("[MoEExpertParallelReduceStage] " << name << " must be FP32");
            return false;
        }
        return true;
    }

    bool matrixShape(const ITensor *tensor, size_t &rows, size_t &cols, const char *name)
    {
        const auto &shape = tensor->shape();
        if (shape.size() != 2)
        {
            LOG_ERROR("[MoEExpertParallelReduceStage] " << name << " must be 2D [rows, cols], got rank " << shape.size());
            return false;
        }
        if (shape[0] == 0 || shape[1] == 0)
        {
            LOG_ERROR("[MoEExpertParallelReduceStage] " << name << " has invalid empty shape ["
                                                        << shape[0] << ", " << shape[1] << "]");
            return false;
        }
        rows = shape[0];
        cols = shape[1];
        return true;
    }

    TensorBase *tensorBase(ITensor *tensor)
    {
        return dynamic_cast<TensorBase *>(tensor);
    }

    const TensorBase *tensorBase(const ITensor *tensor)
    {
        return dynamic_cast<const TensorBase *>(tensor);
    }

    std::string deviceString(DeviceId device)
    {
        return device.is_valid() ? device.to_string() : "invalid";
    }

    MoEExpertParallelReducePartialInfo partialInfoFor(
        const MoEExpertParallelReduceStage::Params &params,
        size_t partial_index,
        const ITensor *partial)
    {
        if (partial_index < params.runtime_partial_results.size())
        {
            const auto &runtime_result = params.runtime_partial_results[partial_index];
            if (runtime_result && runtime_result->ok && runtime_result->partial_tensor == partial &&
                !runtime_result->partial_info.name.empty())
            {
                return runtime_result->partial_info;
            }
        }

        if (partial_index < params.partial_infos.size())
            return params.partial_infos[partial_index];

        MoEExpertParallelReducePartialInfo info;
        info.name = "partial" + std::to_string(partial_index);
        if (const auto *base = tensorBase(partial))
        {
            if (auto current = base->current_device())
                info.source_device = *current;
            else if (base->home_device().is_valid())
                info.source_device = base->home_device();
        }
        return info;
    }

    bool sourceMatchesContinuation(
        const MoEExpertParallelReducePartialInfo &info,
        const std::string &continuation_domain,
        DeviceId continuation_device)
    {
        if (!info.source_domain.empty() && !continuation_domain.empty())
            return info.source_domain == continuation_domain;
        return info.source_device.is_valid() && continuation_device.is_valid() &&
               info.source_device == continuation_device;
    }

    void recordPartialDiagnostics(
        MoEExpertParallelReduceDiagnostics *diagnostics,
        const PartialRuntime &partial,
        const std::string &continuation_domain,
        DeviceId continuation_device,
        MoEExpertParallelReducePartialAccumulationPath accumulation_path)
    {
        if (!diagnostics)
            return;

        MoEExpertParallelReducePartialDiagnostics entry;
        entry.name = partial.info.name;
        entry.source_domain = partial.info.source_domain;
        entry.source_device = partial.info.source_device;
        entry.bytes = partial.bytes;
        entry.host_sync_required = partial.host_sync_required;
        entry.source_is_continuation = sourceMatchesContinuation(
            partial.info,
            continuation_domain,
            continuation_device);
        entry.is_sparse = partial.is_sparse;
        entry.sparse_row_count = partial.sparse_row_count;
        entry.accumulation_path = accumulation_path;
        diagnostics->partials.push_back(std::move(entry));
        if (partial.is_sparse)
            ++diagnostics->sparse_partial_count;
    }

    void estimateOptimizedTransferBytes(
        MoEExpertParallelReduceDiagnostics *diagnostics,
        const PartialRuntime &partial,
        const TensorBase *partial_base,
        DeviceId continuation_device,
        bool already_on_continuation)
    {
        if (!diagnostics || already_on_continuation || !partial_base)
            return;

        const auto current_device = partial_base->current_device();
        if (current_device && current_device->is_gpu() && *current_device != continuation_device)
        {
            if (!partial_base->hostValid())
                diagnostics->device_to_host_bytes += partial.bytes;
            diagnostics->host_to_device_bytes += partial.bytes;
            return;
        }

        diagnostics->host_to_device_bytes += partial.bytes;
    }

    bool validateSelectedRows(
        const std::vector<int> &selected_rows,
        size_t live_rows,
        size_t partial_index)
    {
        std::vector<unsigned char> seen(live_rows, 0);
        for (int row : selected_rows)
        {
            if (row < 0 || static_cast<size_t>(row) >= live_rows)
            {
                LOG_ERROR("[MoEExpertParallelReduceStage] Sparse partial tensor " << partial_index
                                                                                  << " selected row " << row
                                                                                  << " is outside live row range [0, "
                                                                                  << live_rows << ")");
                return false;
            }

            auto &was_seen = seen[static_cast<size_t>(row)];
            if (was_seen)
            {
                LOG_ERROR("[MoEExpertParallelReduceStage] Sparse partial tensor " << partial_index
                                                                                  << " contains duplicate selected row " << row);
                return false;
            }
            was_seen = 1;
        }
        return true;
    }

    TensorBase *sparseExpansionScratchFor(
        const MoEExpertParallelReduceStage::Params &params,
        size_t sparse_index,
        size_t live_rows,
        size_t live_cols,
        const std::string &partial_name)
    {
        if (sparse_index >= params.sparse_expansion_scratch.size())
        {
            LOG_ERROR("[MoEExpertParallelReduceStage] Sparse optimized partial '"
                      << partial_name << "' requires preallocated dense expansion scratch ["
                      << live_rows << ", " << live_cols << "]; none was provided for sparse partial index "
                      << sparse_index);
            return nullptr;
        }

        TensorBase *scratch = params.sparse_expansion_scratch[sparse_index];
        if (!scratch)
        {
            LOG_ERROR("[MoEExpertParallelReduceStage] Sparse optimized partial '"
                      << partial_name << "' has null dense expansion scratch at sparse partial index "
                      << sparse_index);
            return nullptr;
        }
        if (scratch->native_type() != TensorType::FP32)
        {
            LOG_ERROR("[MoEExpertParallelReduceStage] Sparse optimized partial '"
                      << partial_name << "' expansion scratch must be FP32, got "
                      << scratch->dtype_name());
            return nullptr;
        }

        const auto &shape = scratch->shape();
        if (shape.size() != 2 || shape[0] < live_rows || shape[1] != live_cols)
        {
            LOG_ERROR("[MoEExpertParallelReduceStage] Sparse optimized partial '"
                      << partial_name << "' expansion scratch shape ["
                      << (shape.empty() ? 0 : shape[0]) << ", "
                      << (shape.size() < 2 ? 0 : shape[1])
                      << "] cannot hold live output [" << live_rows << ", " << live_cols << "]");
            return nullptr;
        }
        return scratch;
    }

    void logDiagnosticsIfRequested(const MoEExpertParallelReduceDiagnostics &diagnostics)
    {
        const auto &env = debugEnv();
        if (!env.moe_expert_overlay.transfer_trace && !env.moe_expert_overlay.trace && !env.profile.enabled)
            return;

        LOG_INFO("[MoEExpertParallelReduceStage] mode=" << toString(diagnostics.mode)
                 << " host_staged=" << (diagnostics.host_staged ? "true" : "false")
                 << " continuation_domain=" << diagnostics.continuation_domain
                 << " continuation_device=" << deviceString(diagnostics.continuation_device)
                 << " partials=" << diagnostics.partial_count
                 << " sparse_partials=" << diagnostics.sparse_partial_count
                 << " input_bytes=" << diagnostics.input_bytes
                 << " d2h_bytes=" << diagnostics.device_to_host_bytes
                 << " h2d_bytes=" << diagnostics.host_to_device_bytes
                 << " total_transfer_bytes=" << diagnostics.total_transfer_bytes
                 << " reduce_ms=" << diagnostics.reduce_ms
                 << " output_resident_on_continuation="
                 << (diagnostics.output_resident_on_continuation ? "true" : "false"));

        if (env.moe_expert_overlay.transfer_trace)
        {
            for (const auto &partial : diagnostics.partials)
            {
                LOG_INFO("[MoEExpertParallelReduceStage] partial name=" << partial.name
                         << " source_domain=" << partial.source_domain
                         << " source_device=" << deviceString(partial.source_device)
                         << " bytes=" << partial.bytes
                         << " is_sparse=" << (partial.is_sparse ? "true" : "false")
                         << " sparse_row_count=" << partial.sparse_row_count
                         << " host_sync_required=" << (partial.host_sync_required ? "true" : "false")
                         << " source_is_continuation=" << (partial.source_is_continuation ? "true" : "false")
                         << " accumulation_path=" << toString(partial.accumulation_path));
            }
        }
    }

    bool runHostStagedCorrectnessReduce(
        const MoEExpertParallelReduceStage::Params &params,
        const std::vector<PartialRuntime> &runtime_partials,
        TensorBase *output_base,
        size_t element_count,
        size_t live_cols,
        MoEExpertParallelReduceDiagnostics *diagnostics)
    {
        diagnostics->mode = MoEExpertParallelReduceMode::HostStagedCorrectness;
        diagnostics->host_staged = true;

        float *out = params.output->mutable_data();
        std::fill_n(out, element_count, 0.0f);

        for (const auto &partial : runtime_partials)
        {
            diagnostics->input_bytes += partial.bytes;
            diagnostics->host_staged_read_bytes += partial.bytes;
            if (partial.info.source_device.is_gpu() || partial.host_sync_required)
                diagnostics->device_to_host_bytes += partial.bytes;
            recordPartialDiagnostics(
                diagnostics,
                partial,
                params.continuation_domain,
                params.continuation_device,
                MoEExpertParallelReducePartialAccumulationPath::HostSummedCorrectnessFallback);

            if (const auto *partial_base = tensorBase(partial.tensor))
            {
                if (!const_cast<TensorBase *>(partial_base)->ensureOnHost())
                {
                    LOG_ERROR("[MoEExpertParallelReduceStage] Failed to stage partial '"
                              << partial.info.name << "' from domain '" << partial.info.source_domain
                              << "' on host for reduction");
                    return false;
                }
            }

            const float *data = partial.tensor->data();
            if (partial.is_sparse)
            {
                // Sparse partial: scatter-add selected rows only.
                const auto &selected_rows = partial.info.selected_rows;
                for (size_t row_idx = 0; row_idx < selected_rows.size(); ++row_idx)
                {
                    const int full_row = selected_rows[row_idx];
                    for (size_t col = 0; col < live_cols; ++col)
                        out[static_cast<size_t>(full_row) * live_cols + col] +=
                            data[row_idx * live_cols + col];
                }
            }
            else
            {
                // Dense partial: element-wise accumulation over all live elements.
                for (size_t i = 0; i < element_count; ++i)
                    out[i] += data[i];
            }
        }

        if (params.continuation_device.is_gpu())
        {
            diagnostics->host_to_device_bytes += element_count * sizeof(float);
            if (!output_base->ensureOnDevice(params.continuation_device))
            {
                LOG_ERROR("[MoEExpertParallelReduceStage] Failed to upload reduced output to continuation domain '"
                          << params.continuation_domain << "' on " << params.continuation_device.to_string());
                return false;
            }
            diagnostics->output_resident_on_continuation = output_base->is_on_device(params.continuation_device);
        }
        else
        {
            diagnostics->output_resident_on_continuation = output_base->hostValid();
        }

        return true;
    }

    bool runContinuationDeviceOptimizedReduce(
        const MoEExpertParallelReduceStage::Params &params,
        const std::vector<PartialRuntime> &runtime_partials,
        TensorBase *output_base,
        size_t element_count,
        size_t live_cols,
        MoEExpertParallelReduceDiagnostics *diagnostics,
        void *stream)
    {
        if (!params.continuation_device.is_gpu())
        {
            LOG_DEBUG("[MoEExpertParallelReduceStage] " << toString(params.mode)
                                                        << " requested for CPU continuation; using "
                                                        << toString(MoEExpertParallelReduceMode::HostStagedCorrectness)
                                                        << " bridge path");
            return runHostStagedCorrectnessReduce(params, runtime_partials, output_base, element_count, live_cols, diagnostics);
        }

        IBackend *backend = getBackendFor(params.continuation_device);
        if (!backend)
        {
            LOG_ERROR("[MoEExpertParallelReduceStage] No backend available for continuation device "
                      << params.continuation_device.to_string());
            return false;
        }

        diagnostics->mode = MoEExpertParallelReduceMode::ContinuationDeviceOptimized;
        diagnostics->host_staged = false;

        const size_t live_bytes = element_count * sizeof(float);
        const bool preserves_tail = params.output->numel() > element_count;
        if (preserves_tail)
        {
            diagnostics->host_to_device_bytes += params.output->numel() * sizeof(float);
            if (!output_base->ensureOnDevice(params.continuation_device))
            {
                LOG_ERROR("[MoEExpertParallelReduceStage] Failed to upload output tail-preservation buffer to "
                          << params.continuation_device.to_string());
                return false;
            }
        }
        else if (!output_base->allocateOnDevice(params.continuation_device))
        {
            LOG_ERROR("[MoEExpertParallelReduceStage] Failed to allocate reduced output on continuation device "
                      << params.continuation_device.to_string());
            return false;
        }

        void *output_device = output_base->gpu_data_ptr();
        if (!output_device)
        {
            LOG_ERROR("[MoEExpertParallelReduceStage] Output has no device pointer after continuation allocation");
            return false;
        }

        const int continuation_ordinal = params.continuation_device.gpu_ordinal();
        if (!backend->memset(output_device, 0, live_bytes, continuation_ordinal, stream))
        {
            LOG_ERROR("[MoEExpertParallelReduceStage] Failed to zero live output region on "
                      << params.continuation_device.to_string());
            return false;
        }

        size_t sparse_scratch_index = 0;
        for (const auto &partial : runtime_partials)
        {
            diagnostics->input_bytes += partial.bytes;
            auto *partial_base = const_cast<TensorBase *>(tensorBase(partial.tensor));
            if (!partial_base)
            {
                LOG_ERROR("[MoEExpertParallelReduceStage] Optimized continuation-device reduction requires TensorBase partials");
                return false;
            }

            if (partial.is_sparse)
            {
                // Sparse partial: GPU-native scatter-add is deferred (Bridge Phase 7A).
                // Bridge path: scatter-add into a dense host buffer, then upload and
                // accumulate on the continuation device via vectorAddInplace.
                // This keeps accumulation ownership on the continuation device while
                // using host-staged transport for the sparse-to-dense expand step.
                const auto &selected_rows = partial.info.selected_rows;
                const size_t sparse_element_count = selected_rows.size() * live_cols;

                if (!partial_base->ensureOnHost())
                {
                    LOG_ERROR("[MoEExpertParallelReduceStage] Failed to stage sparse partial '"
                              << partial.info.name << "' on host for scatter-add before continuation-device accumulation");
                    return false;
                }

                TensorBase *dense_scratch = sparseExpansionScratchFor(
                    params,
                    sparse_scratch_index++,
                    element_count / live_cols,
                    live_cols,
                    partial.info.name);
                if (!dense_scratch)
                    return false;

                float *dense_data = dense_scratch->mutable_data();
                std::fill_n(dense_data, element_count, 0.0f);

                const float *sparse_data = partial.tensor->data();
                for (size_t row_idx = 0; row_idx < selected_rows.size(); ++row_idx)
                {
                    const int full_row = selected_rows[row_idx];
                    for (size_t col = 0; col < live_cols; ++col)
                        dense_data[static_cast<size_t>(full_row) * live_cols + col] +=
                            sparse_data[row_idx * live_cols + col];
                }

                // Upload expanded dense scratch to continuation device and accumulate.
                diagnostics->device_to_host_bytes += sparse_element_count * sizeof(float);
                diagnostics->host_to_device_bytes += live_bytes;

                if (!dense_scratch->ensureOnDevice(params.continuation_device))
                {
                    LOG_ERROR("[MoEExpertParallelReduceStage] Failed to upload scatter-expanded sparse partial '"
                              << partial.info.name << "' to " << params.continuation_device.to_string());
                    return false;
                }

                const void *scratch_device = dense_scratch->gpu_data_ptr();
                if (!scratch_device)
                {
                    LOG_ERROR("[MoEExpertParallelReduceStage] Expanded sparse partial '" << partial.info.name
                              << "' has no device pointer after H2D upload");
                    return false;
                }

                if (!backend->vectorAddInplace(output_device, scratch_device, element_count,
                                               sizeof(float), continuation_ordinal, stream))
                {
                    LOG_ERROR("[MoEExpertParallelReduceStage] Device accumulation of scatter-expanded sparse partial '"
                              << partial.info.name << "' failed on " << params.continuation_device.to_string());
                    return false;
                }

                recordPartialDiagnostics(
                    diagnostics,
                    partial,
                    params.continuation_domain,
                    params.continuation_device,
                    MoEExpertParallelReducePartialAccumulationPath::HostStagedThenDeviceAccumulated);
                continue;
            }

            const bool already_on_continuation = partial_base->is_on_device(params.continuation_device);
            estimateOptimizedTransferBytes(
                diagnostics,
                partial,
                partial_base,
                params.continuation_device,
                already_on_continuation);

            const auto current_device = partial_base->current_device();
            if (!already_on_continuation &&
                current_device && current_device->is_gpu() &&
                !partial_base->hostValid())
            {
                if (!partial_base->ensureOnHost())
                {
                    LOG_ERROR("[MoEExpertParallelReduceStage] Failed to stage partial '"
                              << partial.info.name << "' from " << current_device->to_string()
                              << " through host before continuation-device accumulation on "
                              << params.continuation_device.to_string());
                    return false;
                }
            }

            if (!partial_base->ensureOnDevice(params.continuation_device))
            {
                LOG_ERROR("[MoEExpertParallelReduceStage] Failed to make partial '"
                          << partial.info.name << "' available on continuation device "
                          << params.continuation_device.to_string());
                return false;
            }

            const void *partial_device = partial_base->gpu_data_ptr();
            if (!partial_device)
            {
                LOG_ERROR("[MoEExpertParallelReduceStage] Partial '" << partial.info.name
                                                                      << "' has no device pointer after transfer to continuation");
                return false;
            }

            if (!backend->vectorAddInplace(output_device, partial_device, element_count,
                                           sizeof(float), continuation_ordinal, stream))
            {
                LOG_ERROR("[MoEExpertParallelReduceStage] Device accumulation failed for partial '"
                          << partial.info.name << "' on " << params.continuation_device.to_string());
                return false;
            }

            recordPartialDiagnostics(
                diagnostics,
                partial,
                params.continuation_domain,
                params.continuation_device,
                already_on_continuation
                    ? MoEExpertParallelReducePartialAccumulationPath::ContinuationDeviceAccumulated
                    : MoEExpertParallelReducePartialAccumulationPath::HostStagedThenDeviceAccumulated);
        }

        output_base->transitionToWithEvent(
            TensorCoherenceState::DEVICE_AUTHORITATIVE,
            params.continuation_device,
            stream);
        diagnostics->output_resident_on_continuation = output_base->is_on_device(params.continuation_device);
        return true;
    }

} // namespace

    const char *toString(MoEExpertParallelReduceMode mode)
    {
        switch (mode)
        {
        case MoEExpertParallelReduceMode::HostStagedCorrectness:
            return "HostStagedCorrectness";
        case MoEExpertParallelReduceMode::ContinuationDeviceOptimized:
            return "ContinuationDeviceOptimized";
        }
        return "Unknown";
    }

    const char *toString(MoEExpertParallelReducePartialAccumulationPath path)
    {
        switch (path)
        {
        case MoEExpertParallelReducePartialAccumulationPath::ContinuationDeviceAccumulated:
            return "ContinuationDeviceAccumulated";
        case MoEExpertParallelReducePartialAccumulationPath::HostStagedThenDeviceAccumulated:
            return "HostStagedThenDeviceAccumulated";
        case MoEExpertParallelReducePartialAccumulationPath::HostSummedCorrectnessFallback:
            return "HostSummedCorrectnessFallback";
        }
        return "Unknown";
    }

    void MoEExpertParallelReduceDiagnostics::clear()
    {
        mode = MoEExpertParallelReduceMode::HostStagedCorrectness;
        continuation_domain.clear();
        continuation_device = DeviceId::invalid();
        host_staged = true;
        output_resident_on_continuation = false;
        partial_count = 0;
        sparse_partial_count = 0;
        input_bytes = 0;
        host_staged_read_bytes = 0;
        device_to_host_bytes = 0;
        host_to_device_bytes = 0;
        total_transfer_bytes = 0;
        reduce_ms = 0.0;
        partials.clear();
    }

    MoEExpertParallelReduceStage::MoEExpertParallelReduceStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool MoEExpertParallelReduceStage::execute(IDeviceContext *ctx)
    {
        (void)ctx;

        if (!validateFP32Tensor(params_.output, "output"))
            return false;

        TensorBase *output_base = tensorBase(params_.output);
        if (!output_base)
        {
            LOG_ERROR("[MoEExpertParallelReduceStage] Output tensor must derive from TensorBase for explicit continuation-domain coherence");
            return false;
        }

        size_t rows = 0;
        size_t cols = 0;
        if (!matrixShape(params_.output, rows, cols, "output"))
            return false;

        const size_t live_rows = params_.rows > 0 ? params_.rows : rows;
        const size_t live_cols = params_.cols > 0 ? params_.cols : cols;

        if (rows < live_rows || cols != live_cols)
        {
            LOG_ERROR("[MoEExpertParallelReduceStage] Output shape [" << rows << ", " << cols
                                                                       << "] cannot hold live output ["
                                                                       << live_rows << ", " << live_cols << "]");
            return false;
        }

        std::vector<PartialRuntime> runtime_partials;
        runtime_partials.reserve(params_.partials.size());
        for (size_t partial_index = 0; partial_index < params_.partials.size(); ++partial_index)
        {
            const ITensor *partial = params_.partials[partial_index];
            if (partial == params_.output)
            {
                LOG_ERROR("[MoEExpertParallelReduceStage] Partial tensor " << partial_index
                                                                            << " aliases output; inputs must be preserved");
                return false;
            }
            if (!validateFP32Tensor(partial, "partial"))
                return false;

            const MoEExpertParallelReducePartialInfo info = partialInfoFor(params_, partial_index, partial);
            const bool is_sparse = !info.selected_rows.empty();

            size_t partial_rows = 0;
            size_t partial_cols = 0;
            if (!matrixShape(partial, partial_rows, partial_cols, "partial"))
                return false;

            if (is_sparse)
            {
                // Sparse partial: shape must be [selected_rows.size(), live_cols].
                if (partial_rows != info.selected_rows.size() || partial_cols != live_cols)
                {
                    LOG_ERROR("[MoEExpertParallelReduceStage] Sparse partial tensor " << partial_index
                                                                                      << " shape [" << partial_rows << ", " << partial_cols
                                                                                      << "] must equal compact selected-row shape [" << info.selected_rows.size()
                                                                                      << " selected rows with " << live_cols << " cols");
                    return false;
                }
                if (!validateSelectedRows(info.selected_rows, live_rows, partial_index))
                    return false;
            }
            else
            {
                // Dense partial: must cover at least live_rows x live_cols.
                if (partial_rows < live_rows || partial_cols != live_cols)
                {
                    LOG_ERROR("[MoEExpertParallelReduceStage] Partial tensor " << partial_index
                                                                                << " shape [" << partial_rows << ", " << partial_cols
                                                                                << "] cannot provide live output shape [" << live_rows << ", " << live_cols << "]");
                    return false;
                }
            }

            PartialRuntime runtime;
            runtime.tensor = partial;
            runtime.info = std::move(info);
            runtime.is_sparse = is_sparse;
            runtime.sparse_row_count = is_sparse ? runtime.info.selected_rows.size() : 0;
            // For sparse partials, bytes reports the compact (selected rows) payload.
            runtime.bytes = is_sparse
                                ? (runtime.sparse_row_count * live_cols * sizeof(float))
                                : (live_rows > 0 && live_cols > 0 ? live_rows * live_cols * sizeof(float) : kUnknownSize);
            if (const auto *base = tensorBase(partial))
                runtime.host_sync_required = base->needsDownload();
            runtime_partials.push_back(std::move(runtime));
        }

        const size_t element_count = live_rows * live_cols;

        MoEExpertParallelReduceDiagnostics local_diagnostics;
        MoEExpertParallelReduceDiagnostics *diagnostics = params_.diagnostics;
        if (!diagnostics && params_.diagnostics_lifetime)
            diagnostics = params_.diagnostics_lifetime.get();
        if (!diagnostics)
            diagnostics = &local_diagnostics;
        diagnostics->clear();
        diagnostics->mode = params_.mode;
        diagnostics->continuation_domain = params_.continuation_domain;
        diagnostics->continuation_device = params_.continuation_device;
        diagnostics->host_staged = params_.mode == MoEExpertParallelReduceMode::HostStagedCorrectness;
        diagnostics->partial_count = runtime_partials.size();

        const auto start_time = std::chrono::steady_clock::now();

        bool ok = false;
        if (params_.mode == MoEExpertParallelReduceMode::ContinuationDeviceOptimized)
        {
            ok = runContinuationDeviceOptimizedReduce(
                params_, runtime_partials, output_base, element_count, live_cols, diagnostics, gpuStream());
        }
        else
        {
            ok = runHostStagedCorrectnessReduce(
                params_, runtime_partials, output_base, element_count, live_cols, diagnostics);
        }

        const auto end_time = std::chrono::steady_clock::now();
        diagnostics->reduce_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        diagnostics->total_transfer_bytes = diagnostics->device_to_host_bytes + diagnostics->host_to_device_bytes;
        logDiagnosticsIfRequested(*diagnostics);
        MoEExpertOverlayProfiler::recordFinalReduce(params_.layer_idx, *diagnostics);

        if (!ok)
        {
            return false;
        }

        return true;
    }

    size_t MoEExpertParallelReduceStage::estimatedFlops() const
    {
        if (!params_.output)
            return 0;
        const size_t partial_count = params_.partials.size();
        return partial_count == 0 ? 0 : params_.output->numel() * partial_count;
    }

    size_t MoEExpertParallelReduceStage::estimatedMemoryBytes() const
    {
        if (!params_.output)
            return 0;
        const size_t bytes = params_.output->numel() * sizeof(float);
        return bytes * (params_.partials.size() + 1);
    }

    bool MoEExpertParallelReduceStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::CPU ||
               backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    StageBufferRequirements MoEExpertParallelReduceStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        for (const ITensor *partial : params_.partials)
        {
            if (partial)
                reqs.addInput("partial", partial->shape(), toBufferTensorType(partial->native_type()));
        }
        if (params_.output)
            reqs.addOutput("output", params_.output->shape(), toBufferTensorType(params_.output->native_type()));
        return reqs;
    }

    StageBufferContract MoEExpertParallelReduceStage::bufferContract() const
    {
        auto contract = StageBufferContract::build();
        if (params_.output_buffer_id != BufferId::_COUNT)
            contract.addOutput(params_.output_buffer_id);
        return contract;
    }

    StageDumpInfo MoEExpertParallelReduceStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        const size_t rows = params_.rows > 0
                    ? params_.rows
                    : (params_.output && params_.output->shape().size() == 2 ? params_.output->shape()[0] : 0);
        const size_t cols = params_.cols > 0
                    ? params_.cols
                    : (params_.output && params_.output->shape().size() == 2 ? params_.output->shape()[1] : 0);

        for (const ITensor *partial : params_.partials)
        {
            if (partial)
                info.addInput("partial", partial, rows, cols);
        }
        if (params_.output)
            info.addOutput("output", params_.output, rows, cols);
        info.addScalarInt("partial_count", static_cast<int>(params_.partials.size()));
        info.addScalarInt("layer_idx", params_.layer_idx);
        info.addScalarInt("reduce_mode", static_cast<int>(params_.mode));
        info.addScalarBool("host_staged", params_.mode == MoEExpertParallelReduceMode::HostStagedCorrectness);
        info.addScalarBool("has_continuation_domain", !params_.continuation_domain.empty());
        return info;
    }

} // namespace llaminar2
