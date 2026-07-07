/**
 * @file GDNLiveStateAllGatherStage.cpp
 * @brief Implementation of GDN live-state LocalTP allgather handoff.
 */

#include "GDNLiveStateAllGatherStage.h"
#include "../../../collective/ILocalTPContext.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <utility>

#ifdef HAVE_CUDA
extern "C"
{
    bool cudaGDN_compact_modular_conv_state(
        const float *gathered,
        float *full,
        int degree,
        int qk_channels,
        int local_v_channels,
        int full_v_channels,
        int history_len,
        int device_idx,
        void *stream);
}
#endif

#ifdef HAVE_ROCM
extern "C"
{
    bool rocmGDN_compact_modular_conv_state(
        const float *gathered,
        float *full,
        int degree,
        int qk_channels,
        int local_v_channels,
        int full_v_channels,
        int history_len,
        int device_idx,
        void *stream);
}
#endif

namespace llaminar2
{
    GDNLiveStateAllGatherStage::GDNLiveStateAllGatherStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    std::string GDNLiveStateAllGatherStage::localConvBufferName() const
    {
        return std::string(WS_LOCAL_CONV_STATE) + "_layer" + std::to_string(params_.layer_idx);
    }

    std::string GDNLiveStateAllGatherStage::gatheredConvBufferName() const
    {
        return std::string(WS_GATHERED_CONV_STATE) + "_layer" + std::to_string(params_.layer_idx);
    }

    std::string GDNLiveStateAllGatherStage::fullConvBufferName() const
    {
        return std::string(WS_FULL_CONV_STATE) + "_layer" + std::to_string(params_.layer_idx);
    }

    std::string GDNLiveStateAllGatherStage::localRecurrenceBufferName() const
    {
        return std::string(WS_LOCAL_RECURRENCE_STATE) + "_layer" + std::to_string(params_.layer_idx);
    }

    std::string GDNLiveStateAllGatherStage::fullRecurrenceBufferName() const
    {
        return std::string(WS_FULL_RECURRENCE_STATE) + "_layer" + std::to_string(params_.layer_idx);
    }

    bool GDNLiveStateAllGatherStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "GDNLiveStateAllGatherStage"))
            return false;

        return runLiveStateAllGather("graph_execution");
    }

    bool GDNLiveStateAllGatherStage::requiresPostVerifierStatePublication() const
    {
        return true;
    }

    bool GDNLiveStateAllGatherStage::publishPostVerifierStateRestore(void *stream)
    {
        if (params_.device_id.is_gpu() && !stream)
        {
            LOG_ERROR("[GDNLiveStateAllGatherStage] MTP verifier-state publication requires an explicit stream");
            return false;
        }

        void *previous_stream = gpuStream();
        if (stream)
            setGPUStream(stream);

        const bool ok = runLiveStateAllGather("mtp_verifier_state_publication");
        setGPUStream(previous_stream);

        if (ok)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "gdn_live_state_allgather_publications",
                1.0,
                "decode",
                params_.device_id.toString(),
                {{"layer", std::to_string(params_.layer_idx)},
                 {"tp_device_idx", std::to_string(params_.tp_device_idx)}});
        }
        return ok;
    }

    bool GDNLiveStateAllGatherStage::runLiveStateAllGather(const char *context)
    {
        if (!params_.device_id.is_gpu())
        {
            LOG_ERROR("[GDNLiveStateAllGatherStage] GPU device required, got "
                      << params_.device_id.toString());
            return false;
        }
        if (!params_.tp_ctx)
        {
            LOG_ERROR("[GDNLiveStateAllGatherStage] Missing LocalTP context");
            return false;
        }
        if (params_.tp_device_idx < 0 || params_.tp_device_idx >= params_.tp_ctx->degree())
        {
            LOG_ERROR("[GDNLiveStateAllGatherStage] Invalid TP device index "
                      << params_.tp_device_idx << " degree=" << params_.tp_ctx->degree());
            return false;
        }
        if (!bound_workspace_)
        {
            LOG_ERROR("[GDNLiveStateAllGatherStage] Workspace was not bound");
            return false;
        }

        void *stream = gpuStream();
        if (!stream)
        {
            LOG_ERROR("[GDNLiveStateAllGatherStage] Explicit non-null GPU stream is required");
            return false;
        }

        auto gather_conv = [&]() -> bool
        {
            if (params_.local_conv_state_floats <= 0 && params_.full_conv_state_floats <= 0)
                return true;
            if (!params_.conv_kernel)
            {
                LOG_ERROR("[GDNLiveStateAllGatherStage] Missing short-conv kernel");
                return false;
            }
            const int degree = params_.tp_ctx->degree();
            const int gathered_conv_state_floats = params_.local_conv_state_floats * degree;
            if (params_.local_conv_state_floats <= 0 ||
                params_.full_conv_state_floats <= 0)
            {
                LOG_ERROR("[GDNLiveStateAllGatherStage] Invalid conv state sizes"
                          << " local=" << params_.local_conv_state_floats
                          << " full=" << params_.full_conv_state_floats
                          << " degree=" << degree);
                return false;
            }
            if (params_.local_conv_state_floats == params_.full_conv_state_floats)
            {
                /*
                 * Some phase-split paths begin prefill with replicated dense/GDN
                 * weights already installed.  The short-conv kernel is therefore
                 * already full-sized and there is nothing to gather.  Returning
                 * here avoids treating an already-mirrored state as a malformed
                 * TP-local state and, more importantly, avoids allgathering
                 * degree * local floats into a full buffer that is only local
                 * floats wide.
                 */
                if (params_.conv_kernel->stateBytes() !=
                    static_cast<size_t>(params_.full_conv_state_floats) * sizeof(float))
                {
                    LOG_ERROR("[GDNLiveStateAllGatherStage] Short-conv kernel is not full-sized for no-op handoff"
                              << " expected_bytes=" << (static_cast<size_t>(params_.full_conv_state_floats) * sizeof(float))
                              << " actual_bytes=" << params_.conv_kernel->stateBytes());
                    return false;
                }
                return true;
            }
            if (!params_.modular_conv_state &&
                params_.full_conv_state_floats != gathered_conv_state_floats)
            {
                LOG_ERROR("[GDNLiveStateAllGatherStage] Uniform conv state requires full == local * degree"
                          << " local=" << params_.local_conv_state_floats
                          << " full=" << params_.full_conv_state_floats
                          << " degree=" << degree);
                return false;
            }
            if (params_.modular_conv_state)
            {
                const int expected_local =
                    (params_.conv_qk_channels + params_.conv_local_v_channels) *
                    params_.conv_history_len;
                const int expected_full =
                    (params_.conv_qk_channels + params_.conv_full_v_channels) *
                    params_.conv_history_len;
                if (params_.conv_history_len <= 0 ||
                    params_.conv_qk_channels <= 0 ||
                    params_.conv_local_v_channels <= 0 ||
                    params_.conv_full_v_channels <= 0 ||
                    params_.conv_full_v_channels != params_.conv_local_v_channels * degree ||
                    params_.local_conv_state_floats != expected_local ||
                    params_.full_conv_state_floats != expected_full)
                {
                    LOG_ERROR("[GDNLiveStateAllGatherStage] Invalid modular conv state layout"
                              << " history=" << params_.conv_history_len
                              << " qk_channels=" << params_.conv_qk_channels
                              << " local_v_channels=" << params_.conv_local_v_channels
                              << " full_v_channels=" << params_.conv_full_v_channels
                              << " local=" << params_.local_conv_state_floats
                              << " expected_local=" << expected_local
                              << " full=" << params_.full_conv_state_floats
                              << " expected_full=" << expected_full
                              << " degree=" << degree);
                    return false;
                }
                if (params_.full_conv_state_floats >= gathered_conv_state_floats)
                {
                    LOG_ERROR("[GDNLiveStateAllGatherStage] Modular conv state unexpectedly does not need compaction"
                              << " full=" << params_.full_conv_state_floats
                              << " gathered=" << gathered_conv_state_floats);
                    return false;
                }
            }
            else if (params_.conv_history_len != 0 ||
                     params_.conv_qk_channels != 0 ||
                     params_.conv_local_v_channels != 0 ||
                     params_.conv_full_v_channels != 0)
            {
                LOG_ERROR("[GDNLiveStateAllGatherStage] Uniform conv state must not carry modular layout fields");
                return false;
            }
            if (params_.conv_kernel->stateBytes() !=
                static_cast<size_t>(params_.local_conv_state_floats) * sizeof(float))
            {
                LOG_ERROR("[GDNLiveStateAllGatherStage] Short-conv kernel state is not local-sized before handoff"
                          << " expected_bytes=" << (static_cast<size_t>(params_.local_conv_state_floats) * sizeof(float))
                          << " actual_bytes=" << params_.conv_kernel->stateBytes());
                return false;
            }

            const std::string local_name = localConvBufferName();
            const std::string full_name = fullConvBufferName();
            const std::string gathered_name = gatheredConvBufferName();
            auto *local = static_cast<float *>(bound_workspace_->getBuffer(local_name));
            auto *full = static_cast<float *>(bound_workspace_->getBuffer(full_name));
            auto *gathered = params_.modular_conv_state
                                 ? static_cast<float *>(bound_workspace_->getBuffer(gathered_name))
                                 : full;
            if (!local || !full)
            {
                LOG_ERROR("[GDNLiveStateAllGatherStage] Missing conv workspace buffers");
                return false;
            }
            if (params_.modular_conv_state && !gathered)
            {
                LOG_ERROR("[GDNLiveStateAllGatherStage] Missing gathered modular conv workspace buffer");
                return false;
            }
            if (!params_.conv_kernel->exportState(nullptr, local, stream))
            {
                LOG_ERROR("[GDNLiveStateAllGatherStage] Failed to export short-conv state");
                return false;
            }
            if (!params_.tp_ctx->allgatherRawOnStream(
                    local,
                    gathered,
                    static_cast<size_t>(params_.local_conv_state_floats),
                    CollectiveDataType::FLOAT32,
                    params_.tp_device_idx,
                    stream,
                    params_.stage_name + "_conv"))
            {
                LOG_ERROR("[GDNLiveStateAllGatherStage] Short-conv state allgather failed");
                return false;
            }
            if (params_.modular_conv_state)
            {
                bool compact_ok = false;
#ifdef HAVE_CUDA
                if (params_.device_id.is_cuda())
                {
                    compact_ok = cudaGDN_compact_modular_conv_state(
                        gathered,
                        full,
                        degree,
                        params_.conv_qk_channels,
                        params_.conv_local_v_channels,
                        params_.conv_full_v_channels,
                        params_.conv_history_len,
                        params_.device_id.cuda_ordinal(),
                        stream);
                }
#endif
#ifdef HAVE_ROCM
                if (params_.device_id.is_rocm())
                {
                    compact_ok = rocmGDN_compact_modular_conv_state(
                        gathered,
                        full,
                        degree,
                        params_.conv_qk_channels,
                        params_.conv_local_v_channels,
                        params_.conv_full_v_channels,
                        params_.conv_history_len,
                        params_.device_id.rocm_ordinal(),
                        stream);
                }
#endif
                if (!compact_ok)
                {
                    LOG_ERROR("[GDNLiveStateAllGatherStage] Modular short-conv state compaction failed");
                    return false;
                }
            }
            params_.conv_kernel->allocateGPUState(params_.full_conv_state_floats);
            if (params_.conv_kernel->stateBytes() !=
                static_cast<size_t>(params_.full_conv_state_floats) * sizeof(float))
            {
                LOG_ERROR("[GDNLiveStateAllGatherStage] Short-conv full-state allocation failed");
                return false;
            }
            if (!params_.conv_kernel->importState(nullptr, full, stream))
            {
                LOG_ERROR("[GDNLiveStateAllGatherStage] Failed to import full short-conv state");
                return false;
            }
            return true;
        };

        auto gather_recurrence = [&]() -> bool
        {
            if (params_.local_recurrence_state_floats <= 0 &&
                params_.full_recurrence_state_floats <= 0)
                return true;
            if (!params_.recurrence_kernel)
            {
                LOG_ERROR("[GDNLiveStateAllGatherStage] Missing recurrence kernel");
                return false;
            }
            if (params_.local_recurrence_state_floats > 0 &&
                params_.full_recurrence_state_floats > 0 &&
                params_.local_recurrence_state_floats ==
                    params_.full_recurrence_state_floats)
            {
                /*
                 * Matching local/full recurrence sizes mean the graph is already
                 * operating on mirrored dense state.  There is no TP-local
                 * recurrence bank to gather, so preserve the current kernel
                 * state and let downstream replicated decode consume it.
                 */
                if (params_.recurrence_kernel->stateBytes() !=
                    static_cast<size_t>(params_.full_recurrence_state_floats) * sizeof(float))
                {
                    LOG_ERROR("[GDNLiveStateAllGatherStage] Recurrence kernel is not full-sized for no-op handoff"
                              << " expected_bytes=" << (static_cast<size_t>(params_.full_recurrence_state_floats) * sizeof(float))
                              << " actual_bytes=" << params_.recurrence_kernel->stateBytes());
                    return false;
                }
                return true;
            }
            if (params_.local_recurrence_state_floats <= 0 ||
                params_.full_recurrence_state_floats <= 0 ||
                params_.full_recurrence_state_floats !=
                    params_.local_recurrence_state_floats * params_.tp_ctx->degree())
            {
                LOG_ERROR("[GDNLiveStateAllGatherStage] Invalid recurrence state sizes"
                          << " local=" << params_.local_recurrence_state_floats
                          << " full=" << params_.full_recurrence_state_floats
                          << " degree=" << params_.tp_ctx->degree());
                return false;
            }
            if (params_.recurrence_kernel->stateBytes() !=
                static_cast<size_t>(params_.local_recurrence_state_floats) * sizeof(float))
            {
                LOG_ERROR("[GDNLiveStateAllGatherStage] Recurrence kernel state is not local-sized before handoff"
                          << " expected_bytes=" << (static_cast<size_t>(params_.local_recurrence_state_floats) * sizeof(float))
                          << " actual_bytes=" << params_.recurrence_kernel->stateBytes());
                return false;
            }

            const std::string local_name = localRecurrenceBufferName();
            const std::string full_name = fullRecurrenceBufferName();
            auto *local = static_cast<float *>(bound_workspace_->getBuffer(local_name));
            auto *full = static_cast<float *>(bound_workspace_->getBuffer(full_name));
            if (!local || !full)
            {
                LOG_ERROR("[GDNLiveStateAllGatherStage] Missing recurrence workspace buffers");
                return false;
            }
            if (!params_.recurrence_kernel->exportState(nullptr, local, stream))
            {
                LOG_ERROR("[GDNLiveStateAllGatherStage] Failed to export recurrence state");
                return false;
            }
            if (!params_.tp_ctx->allgatherRawOnStream(
                    local,
                    full,
                    static_cast<size_t>(params_.local_recurrence_state_floats),
                    CollectiveDataType::FLOAT32,
                    params_.tp_device_idx,
                    stream,
                    params_.stage_name + "_recurrence"))
            {
                LOG_ERROR("[GDNLiveStateAllGatherStage] Recurrence state allgather failed");
                return false;
            }
            params_.recurrence_kernel->allocateGPUState(params_.full_recurrence_state_floats);
            if (params_.recurrence_kernel->stateBytes() !=
                static_cast<size_t>(params_.full_recurrence_state_floats) * sizeof(float))
            {
                LOG_ERROR("[GDNLiveStateAllGatherStage] Recurrence full-state allocation failed");
                return false;
            }
            if (!params_.recurrence_kernel->importState(nullptr, full, stream))
            {
                LOG_ERROR("[GDNLiveStateAllGatherStage] Failed to import full recurrence state");
                return false;
            }
            return true;
        };

        const bool ok = gather_conv() && gather_recurrence();
        if (ok)
        {
            LOG_DEBUG("[GDNLiveStateAllGatherStage] Gathered full GDN live state"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.toString()
                      << " tp_device_idx=" << params_.tp_device_idx
                      << " context=" << (context ? context : "unknown"));
        }
        return ok;
    }

    size_t GDNLiveStateAllGatherStage::estimatedMemoryBytes() const
    {
        const int gathered_conv_state_floats =
            params_.modular_conv_state && params_.tp_ctx
                ? params_.local_conv_state_floats * params_.tp_ctx->degree()
                : 0;
        return (static_cast<size_t>(std::max(params_.local_conv_state_floats, 0)) +
                static_cast<size_t>(std::max(gathered_conv_state_floats, 0)) +
                static_cast<size_t>(std::max(params_.full_conv_state_floats, 0)) +
                static_cast<size_t>(std::max(params_.local_recurrence_state_floats, 0)) +
                static_cast<size_t>(std::max(params_.full_recurrence_state_floats, 0))) *
               sizeof(float);
    }

    bool GDNLiveStateAllGatherStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
#ifdef HAVE_CUDA
        case ComputeBackendType::GPU_CUDA:
            return true;
#endif
#ifdef HAVE_ROCM
        case ComputeBackendType::GPU_ROCM:
            return true;
#endif
        default:
            return false;
        }
    }

    bool GDNLiveStateAllGatherStage::isGraphCapturable() const
    {
        if (!params_.device_id.is_gpu() || !params_.tp_ctx)
            return false;
        bool backend_supported = false;
#ifdef HAVE_CUDA
        backend_supported = backend_supported || params_.device_id.is_cuda();
#endif
#ifdef HAVE_ROCM
        backend_supported = backend_supported || params_.device_id.is_rocm();
#endif
        if (!backend_supported)
            return false;
        if (params_.tp_device_idx < 0 || params_.tp_device_idx >= params_.tp_ctx->degree())
            return false;
        if (params_.tp_ctx->degree() <= 1)
            return false;
        if (params_.local_conv_state_floats < 0 ||
            params_.full_conv_state_floats < 0 ||
            params_.local_recurrence_state_floats < 0 ||
            params_.full_recurrence_state_floats < 0)
            return false;
        if ((params_.local_conv_state_floats > 0 || params_.full_conv_state_floats > 0) &&
            !params_.conv_kernel)
            return false;
        if ((params_.local_recurrence_state_floats > 0 || params_.full_recurrence_state_floats > 0) &&
            !params_.recurrence_kernel)
            return false;
        return params_.tp_ctx->supportsRawAllgatherOnStreamGraphCapture();
    }

    StageDumpInfo GDNLiveStateAllGatherStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        info.addScalarInt("layer_idx", params_.layer_idx)
            .addScalarInt("tp_device_idx", params_.tp_device_idx)
            .addScalarInt("local_conv_state_floats", params_.local_conv_state_floats)
            .addScalarInt("full_conv_state_floats", params_.full_conv_state_floats)
            .addScalarBool("modular_conv_state", params_.modular_conv_state)
            .addScalarInt("conv_history_len", params_.conv_history_len)
            .addScalarInt("conv_qk_channels", params_.conv_qk_channels)
            .addScalarInt("conv_local_v_channels", params_.conv_local_v_channels)
            .addScalarInt("conv_full_v_channels", params_.conv_full_v_channels)
            .addScalarInt("local_recurrence_state_floats", params_.local_recurrence_state_floats)
            .addScalarInt("full_recurrence_state_floats", params_.full_recurrence_state_floats);
        return info;
    }

    StageBufferRequirements GDNLiveStateAllGatherStage::getBufferRequirements() const
    {
        return {};
    }

    StageBufferContract GDNLiveStateAllGatherStage::bufferContract() const
    {
        return {};
    }

    WorkspaceRequirements GDNLiveStateAllGatherStage::getWorkspaceRequirements(int m, int n, int k) const
    {
        (void)m;
        (void)n;
        (void)k;
        WorkspaceRequirements reqs;
        if (params_.local_conv_state_floats > 0)
        {
            reqs.buffers.push_back({localConvBufferName(),
                                    static_cast<size_t>(params_.local_conv_state_floats) * sizeof(float),
                                    256,
                                    true});
        }
        if (params_.modular_conv_state && params_.tp_ctx && params_.local_conv_state_floats > 0)
        {
            reqs.buffers.push_back({gatheredConvBufferName(),
                                    static_cast<size_t>(params_.local_conv_state_floats) *
                                        static_cast<size_t>(params_.tp_ctx->degree()) *
                                        sizeof(float),
                                    256,
                                    true});
        }
        if (params_.full_conv_state_floats > 0)
        {
            reqs.buffers.push_back({fullConvBufferName(),
                                    static_cast<size_t>(params_.full_conv_state_floats) * sizeof(float),
                                    256,
                                    true});
        }
        if (params_.local_recurrence_state_floats > 0)
        {
            reqs.buffers.push_back({localRecurrenceBufferName(),
                                    static_cast<size_t>(params_.local_recurrence_state_floats) * sizeof(float),
                                    256,
                                    true});
        }
        if (params_.full_recurrence_state_floats > 0)
        {
            reqs.buffers.push_back({fullRecurrenceBufferName(),
                                    static_cast<size_t>(params_.full_recurrence_state_floats) * sizeof(float),
                                    256,
                                    true});
        }
        return reqs;
    }

    void GDNLiveStateAllGatherStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        bound_workspace_ = workspace;
    }

    void GDNLiveStateAllGatherStage::unbindWorkspace()
    {
        bound_workspace_ = nullptr;
    }
} // namespace llaminar2
