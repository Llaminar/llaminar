/**
 * @file GDNLiveStateLocalizeStage.cpp
 * @brief Implementation of the mirrored-to-TP-local GDN live-state handoff.
 *
 * Dense-replicated LocalTP decode keeps a full GDN short-conv and recurrence
 * state bank on every device.  The grouped MTP verifier intentionally returns
 * to the economical TP-local GDN graph, so every verifier replay must begin by
 * slicing that mirrored bank back down to the participant-local bank.  This
 * stage performs that slice on the explicit graph stream before any GDN stage
 * reads live state.
 */

#include "GDNLiveStateLocalizeStage.h"
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
    void cudaGDN_gpu_memcpy_async(float *dst, const float *src, size_t count, void *stream);

    bool cudaGDN_slice_modular_conv_state(
        const float *full,
        float *local,
        int rank,
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
    void rocmGDN_gpu_memcpy_async(float *dst, const float *src, size_t count, void *stream);

    bool rocmGDN_slice_modular_conv_state(
        const float *full,
        float *local,
        int rank,
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
    GDNLiveStateLocalizeStage::GDNLiveStateLocalizeStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    std::string GDNLiveStateLocalizeStage::fullConvBufferName() const
    {
        return std::string(WS_FULL_CONV_STATE) + "_layer" + std::to_string(params_.layer_idx);
    }

    std::string GDNLiveStateLocalizeStage::localConvBufferName() const
    {
        return std::string(WS_LOCAL_CONV_STATE) + "_layer" + std::to_string(params_.layer_idx);
    }

    std::string GDNLiveStateLocalizeStage::fullRecurrenceBufferName() const
    {
        return std::string(WS_FULL_RECURRENCE_STATE) + "_layer" + std::to_string(params_.layer_idx);
    }

    std::string GDNLiveStateLocalizeStage::localRecurrenceBufferName() const
    {
        return std::string(WS_LOCAL_RECURRENCE_STATE) + "_layer" + std::to_string(params_.layer_idx);
    }

    bool GDNLiveStateLocalizeStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "GDNLiveStateLocalizeStage"))
            return false;

        return runLiveStateLocalize("graph_execution");
    }

    bool GDNLiveStateLocalizeStage::runLiveStateLocalize(const char *context)
    {
        if (!params_.device_id.is_gpu())
        {
            LOG_ERROR("[GDNLiveStateLocalizeStage] GPU device required, got "
                      << params_.device_id.toString());
            return false;
        }
        if (!params_.tp_ctx)
        {
            LOG_ERROR("[GDNLiveStateLocalizeStage] Missing LocalTP context");
            return false;
        }
        const int degree = params_.tp_ctx->degree();
        if (degree <= 1)
        {
            LOG_ERROR("[GDNLiveStateLocalizeStage] LocalTP degree must be greater than one"
                      << " degree=" << degree);
            return false;
        }
        if (params_.tp_device_idx < 0 || params_.tp_device_idx >= degree)
        {
            LOG_ERROR("[GDNLiveStateLocalizeStage] Invalid TP device index "
                      << params_.tp_device_idx << " degree=" << degree);
            return false;
        }
        if (!bound_workspace_)
        {
            LOG_ERROR("[GDNLiveStateLocalizeStage] Workspace was not bound");
            return false;
        }

        void *stream = gpuStream();
        if (!stream)
        {
            LOG_ERROR("[GDNLiveStateLocalizeStage] Explicit non-null GPU stream is required");
            return false;
        }

        auto copy_contiguous_slice = [&](float *local,
                                         const float *full,
                                         int local_floats,
                                         const char *state_kind) -> bool
        {
            if (!local || !full || local_floats <= 0)
            {
                LOG_ERROR("[GDNLiveStateLocalizeStage] Invalid contiguous slice inputs for "
                          << state_kind);
                return false;
            }
            const float *src =
                full + static_cast<size_t>(params_.tp_device_idx) *
                           static_cast<size_t>(local_floats);
#ifdef HAVE_CUDA
            if (params_.device_id.is_cuda())
            {
                cudaGDN_gpu_memcpy_async(local, src, static_cast<size_t>(local_floats), stream);
                return true;
            }
#endif
#ifdef HAVE_ROCM
            if (params_.device_id.is_rocm())
            {
                rocmGDN_gpu_memcpy_async(local, src, static_cast<size_t>(local_floats), stream);
                return true;
            }
#endif
            LOG_ERROR("[GDNLiveStateLocalizeStage] Unsupported backend for contiguous "
                      << state_kind << " state slice: " << params_.device_id.toString());
            return false;
        };

        auto localize_conv = [&]() -> bool
        {
            if (params_.local_conv_state_floats <= 0 &&
                params_.full_conv_state_floats <= 0)
                return true;
            if (!params_.conv_kernel)
            {
                LOG_ERROR("[GDNLiveStateLocalizeStage] Missing short-conv kernel");
                return false;
            }
            if (params_.local_conv_state_floats <= 0 ||
                params_.full_conv_state_floats <= 0)
            {
                LOG_ERROR("[GDNLiveStateLocalizeStage] Invalid conv state sizes"
                          << " local=" << params_.local_conv_state_floats
                          << " full=" << params_.full_conv_state_floats);
                return false;
            }
            if (params_.local_conv_state_floats == params_.full_conv_state_floats)
            {
                /*
                 * Equal local/full sizes are already mirrored by construction.
                 * The graph normally skips this stage in that case, but the
                 * check keeps manually constructed tests and future graph
                 * variants honest.
                 */
                if (params_.conv_kernel->stateBytes() !=
                    static_cast<size_t>(params_.full_conv_state_floats) * sizeof(float))
                {
                    LOG_ERROR("[GDNLiveStateLocalizeStage] Short-conv no-op localization expected full state"
                              << " expected_bytes=" << (static_cast<size_t>(params_.full_conv_state_floats) * sizeof(float))
                              << " actual_bytes=" << params_.conv_kernel->stateBytes());
                    return false;
                }
                return true;
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
                    LOG_ERROR("[GDNLiveStateLocalizeStage] Invalid modular conv state layout"
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
            }
            else
            {
                if (params_.full_conv_state_floats !=
                    params_.local_conv_state_floats * degree)
                {
                    LOG_ERROR("[GDNLiveStateLocalizeStage] Uniform conv state requires full == local * degree"
                              << " local=" << params_.local_conv_state_floats
                              << " full=" << params_.full_conv_state_floats
                              << " degree=" << degree);
                    return false;
                }
                if (params_.conv_history_len != 0 ||
                    params_.conv_qk_channels != 0 ||
                    params_.conv_local_v_channels != 0 ||
                    params_.conv_full_v_channels != 0)
                {
                    LOG_ERROR("[GDNLiveStateLocalizeStage] Uniform conv state must not carry modular layout fields");
                    return false;
                }
            }

            /*
             * This stage exists only when the verifier graph starts from the
             * replicated decode state bank.  If a local-sized state reaches this
             * point, the lifecycle is wrong; returning success would make graph
             * capture bake in a no-op and then replay stale local state.
             */
            if (params_.conv_kernel->stateBytes() !=
                static_cast<size_t>(params_.full_conv_state_floats) * sizeof(float))
            {
                LOG_ERROR("[GDNLiveStateLocalizeStage] Short-conv kernel state is not full-sized before localization"
                          << " expected_bytes=" << (static_cast<size_t>(params_.full_conv_state_floats) * sizeof(float))
                          << " actual_bytes=" << params_.conv_kernel->stateBytes());
                return false;
            }

            auto *full = static_cast<float *>(bound_workspace_->getBuffer(fullConvBufferName()));
            auto *local = static_cast<float *>(bound_workspace_->getBuffer(localConvBufferName()));
            if (!full || !local)
            {
                LOG_ERROR("[GDNLiveStateLocalizeStage] Missing conv workspace buffers");
                return false;
            }
            if (!params_.conv_kernel->exportState(nullptr, full, stream))
            {
                LOG_ERROR("[GDNLiveStateLocalizeStage] Failed to export full short-conv state");
                return false;
            }

            bool slice_ok = false;
            if (params_.modular_conv_state)
            {
#ifdef HAVE_CUDA
                if (params_.device_id.is_cuda())
                {
                    slice_ok = cudaGDN_slice_modular_conv_state(
                        full,
                        local,
                        params_.tp_device_idx,
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
                    slice_ok = rocmGDN_slice_modular_conv_state(
                        full,
                        local,
                        params_.tp_device_idx,
                        degree,
                        params_.conv_qk_channels,
                        params_.conv_local_v_channels,
                        params_.conv_full_v_channels,
                        params_.conv_history_len,
                        params_.device_id.rocm_ordinal(),
                        stream);
                }
#endif
                if (!slice_ok)
                {
                    LOG_ERROR("[GDNLiveStateLocalizeStage] Modular short-conv state slice failed");
                    return false;
                }
            }
            else if (!copy_contiguous_slice(
                         local,
                         full,
                         params_.local_conv_state_floats,
                         "short-conv"))
            {
                return false;
            }

            if (!params_.conv_kernel->importStateForSize(
                    params_.local_conv_state_floats,
                    nullptr,
                    local,
                    stream))
            {
                LOG_ERROR("[GDNLiveStateLocalizeStage] Failed to import TP-local short-conv state");
                return false;
            }
            return true;
        };

        auto localize_recurrence = [&]() -> bool
        {
            if (params_.local_recurrence_state_floats <= 0 &&
                params_.full_recurrence_state_floats <= 0)
                return true;
            if (!params_.recurrence_kernel)
            {
                LOG_ERROR("[GDNLiveStateLocalizeStage] Missing recurrence kernel");
                return false;
            }
            if (params_.local_recurrence_state_floats <= 0 ||
                params_.full_recurrence_state_floats <= 0)
            {
                LOG_ERROR("[GDNLiveStateLocalizeStage] Invalid recurrence state sizes"
                          << " local=" << params_.local_recurrence_state_floats
                          << " full=" << params_.full_recurrence_state_floats);
                return false;
            }
            if (params_.local_recurrence_state_floats ==
                params_.full_recurrence_state_floats)
            {
                if (params_.recurrence_kernel->stateBytes() !=
                    static_cast<size_t>(params_.full_recurrence_state_floats) * sizeof(float))
                {
                    LOG_ERROR("[GDNLiveStateLocalizeStage] Recurrence no-op localization expected full state"
                              << " expected_bytes=" << (static_cast<size_t>(params_.full_recurrence_state_floats) * sizeof(float))
                              << " actual_bytes=" << params_.recurrence_kernel->stateBytes());
                    return false;
                }
                return true;
            }
            if (params_.full_recurrence_state_floats !=
                params_.local_recurrence_state_floats * degree)
            {
                LOG_ERROR("[GDNLiveStateLocalizeStage] Recurrence state requires full == local * degree"
                          << " local=" << params_.local_recurrence_state_floats
                          << " full=" << params_.full_recurrence_state_floats
                          << " degree=" << degree);
                return false;
            }
            if (params_.recurrence_kernel->stateBytes() !=
                static_cast<size_t>(params_.full_recurrence_state_floats) * sizeof(float))
            {
                LOG_ERROR("[GDNLiveStateLocalizeStage] Recurrence kernel state is not full-sized before localization"
                          << " expected_bytes=" << (static_cast<size_t>(params_.full_recurrence_state_floats) * sizeof(float))
                          << " actual_bytes=" << params_.recurrence_kernel->stateBytes());
                return false;
            }

            auto *full = static_cast<float *>(bound_workspace_->getBuffer(fullRecurrenceBufferName()));
            auto *local = static_cast<float *>(bound_workspace_->getBuffer(localRecurrenceBufferName()));
            if (!full || !local)
            {
                LOG_ERROR("[GDNLiveStateLocalizeStage] Missing recurrence workspace buffers");
                return false;
            }
            if (!params_.recurrence_kernel->exportState(nullptr, full, stream))
            {
                LOG_ERROR("[GDNLiveStateLocalizeStage] Failed to export full recurrence state");
                return false;
            }
            if (!copy_contiguous_slice(
                    local,
                    full,
                    params_.local_recurrence_state_floats,
                    "recurrence"))
            {
                return false;
            }
            if (!params_.recurrence_kernel->importStateForSize(
                    params_.local_recurrence_state_floats,
                    nullptr,
                    local,
                    stream))
            {
                LOG_ERROR("[GDNLiveStateLocalizeStage] Failed to import TP-local recurrence state");
                return false;
            }
            return true;
        };

        const bool ok = localize_conv() && localize_recurrence();
        if (ok)
        {
            LOG_DEBUG("[GDNLiveStateLocalizeStage] Localized mirrored GDN live state"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.toString()
                      << " tp_device_idx=" << params_.tp_device_idx
                      << " context=" << (context ? context : "unknown"));
            PerfStatsCollector::addCounter(
                "mtp",
                "gdn_live_state_localize_runs",
                1.0,
                "decode",
                params_.device_id.toString(),
                {{"layer", std::to_string(params_.layer_idx)},
                 {"tp_device_idx", std::to_string(params_.tp_device_idx)}});
        }
        return ok;
    }

    size_t GDNLiveStateLocalizeStage::estimatedMemoryBytes() const
    {
        return (static_cast<size_t>(std::max(params_.local_conv_state_floats, 0)) +
                static_cast<size_t>(std::max(params_.full_conv_state_floats, 0)) +
                static_cast<size_t>(std::max(params_.local_recurrence_state_floats, 0)) +
                static_cast<size_t>(std::max(params_.full_recurrence_state_floats, 0))) *
               sizeof(float);
    }

    bool GDNLiveStateLocalizeStage::supportsBackend(ComputeBackendType backend) const
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

    bool GDNLiveStateLocalizeStage::isGraphCapturable() const
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
        if (params_.tp_ctx->degree() <= 1 ||
            params_.tp_device_idx < 0 ||
            params_.tp_device_idx >= params_.tp_ctx->degree())
            return false;
        if (params_.local_conv_state_floats < 0 ||
            params_.full_conv_state_floats < 0 ||
            params_.local_recurrence_state_floats < 0 ||
            params_.full_recurrence_state_floats < 0)
            return false;
        if ((params_.local_conv_state_floats > 0 ||
             params_.full_conv_state_floats > 0) &&
            !params_.conv_kernel)
            return false;
        if ((params_.local_recurrence_state_floats > 0 ||
             params_.full_recurrence_state_floats > 0) &&
            !params_.recurrence_kernel)
            return false;
        return true;
    }

    StageDumpInfo GDNLiveStateLocalizeStage::buildDumpInfoImpl() const
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

    StageBufferRequirements GDNLiveStateLocalizeStage::getBufferRequirements() const
    {
        return {};
    }

    StageBufferContract GDNLiveStateLocalizeStage::bufferContract() const
    {
        return {};
    }

    WorkspaceRequirements GDNLiveStateLocalizeStage::getWorkspaceRequirements(int m, int n, int k) const
    {
        (void)m;
        (void)n;
        (void)k;
        WorkspaceRequirements reqs;
        if (params_.local_conv_state_floats > 0 &&
            params_.full_conv_state_floats > 0 &&
            params_.local_conv_state_floats != params_.full_conv_state_floats)
        {
            reqs.buffers.push_back({fullConvBufferName(),
                                    static_cast<size_t>(params_.full_conv_state_floats) * sizeof(float),
                                    256,
                                    true});
            reqs.buffers.push_back({localConvBufferName(),
                                    static_cast<size_t>(params_.local_conv_state_floats) * sizeof(float),
                                    256,
                                    true});
        }
        if (params_.local_recurrence_state_floats > 0 &&
            params_.full_recurrence_state_floats > 0 &&
            params_.local_recurrence_state_floats != params_.full_recurrence_state_floats)
        {
            reqs.buffers.push_back({fullRecurrenceBufferName(),
                                    static_cast<size_t>(params_.full_recurrence_state_floats) * sizeof(float),
                                    256,
                                    true});
            reqs.buffers.push_back({localRecurrenceBufferName(),
                                    static_cast<size_t>(params_.local_recurrence_state_floats) * sizeof(float),
                                    256,
                                    true});
        }
        return reqs;
    }

    void GDNLiveStateLocalizeStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        bound_workspace_ = workspace;
    }

    void GDNLiveStateLocalizeStage::unbindWorkspace()
    {
        bound_workspace_ = nullptr;
    }
} // namespace llaminar2
