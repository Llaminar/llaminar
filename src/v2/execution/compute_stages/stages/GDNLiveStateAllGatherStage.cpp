/**
 * @file GDNLiveStateAllGatherStage.cpp
 * @brief Captured modulo-linked GDN live-state allgather implementation.
 *
 * A raw equal-count collective produces `[rank0 groups | rank1 groups | ...]`.
 * Full mirrored decode consumes `[Q all ranks | K all ranks | V repeat 0 all
 * ranks | ...]`. The stage retains distinct gathered and full buffers so the
 * device permutation is race-free and graph replay keeps every address stable.
 */

#include "GDNLiveStateAllGatherStage.h"
#include "../../../collective/ILocalTPContext.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../kernels/common/GDNLinkedStateLayout.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../utils/Logger.h"

#include <utility>

namespace llaminar2
{
    GDNLiveStateAllGatherStage::GDNLiveStateAllGatherStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    std::string GDNLiveStateAllGatherStage::localConvBufferName() const
    {
        return std::string(WS_LOCAL_CONV_STATE) + "_layer" +
               std::to_string(params_.layer_idx);
    }

    std::string GDNLiveStateAllGatherStage::gatheredConvBufferName() const
    {
        return std::string(WS_GATHERED_CONV_STATE) + "_layer" +
               std::to_string(params_.layer_idx);
    }

    std::string GDNLiveStateAllGatherStage::fullConvBufferName() const
    {
        return std::string(WS_FULL_CONV_STATE) + "_layer" +
               std::to_string(params_.layer_idx);
    }

    std::string GDNLiveStateAllGatherStage::localRecurrenceBufferName() const
    {
        return std::string(WS_LOCAL_RECURRENCE_STATE) + "_layer" +
               std::to_string(params_.layer_idx);
    }

    std::string GDNLiveStateAllGatherStage::gatheredRecurrenceBufferName() const
    {
        return std::string(WS_GATHERED_RECURRENCE_STATE) + "_layer" +
               std::to_string(params_.layer_idx);
    }

    std::string GDNLiveStateAllGatherStage::fullRecurrenceBufferName() const
    {
        return std::string(WS_FULL_RECURRENCE_STATE) + "_layer" +
               std::to_string(params_.layer_idx);
    }

    bool GDNLiveStateAllGatherStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "GDNLiveStateAllGatherStage"))
            return false;
        return runLiveStateAllGather("graph_execution");
    }

    bool GDNLiveStateAllGatherStage::reassembleOnDevice(
        const float *gathered,
        float *full,
        const GDNLinkedLiveStateShape &shape,
        void *stream) const
    {
#ifdef HAVE_CUDA
        if (params_.device_id.is_cuda())
        {
            return cudaGDN_reassemble_modulo_linked_state(
                gathered,
                full,
                shape.degree,
                shape.global_key_heads,
                params_.geometry.global_value_heads,
                shape.key_elements_per_head,
                shape.value_elements_per_head,
                shape.prefix_group_count,
                params_.device_id.cuda_ordinal(),
                stream);
        }
#endif
#ifdef HAVE_ROCM
        if (params_.device_id.is_rocm())
        {
            return rocmGDN_reassemble_modulo_linked_state(
                gathered,
                full,
                shape.degree,
                shape.global_key_heads,
                params_.geometry.global_value_heads,
                shape.key_elements_per_head,
                shape.value_elements_per_head,
                shape.prefix_group_count,
                params_.device_id.rocm_ordinal(),
                stream);
        }
#endif
        return false;
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
        const int degree = params_.tp_ctx->degree();
        if (degree <= 1 || params_.tp_device_idx < 0 ||
            params_.tp_device_idx >= degree)
        {
            LOG_ERROR("[GDNLiveStateAllGatherStage] Invalid TP participant"
                      << " participant=" << params_.tp_device_idx
                      << " degree=" << degree);
            return false;
        }
        if (!bound_workspace_)
        {
            LOG_ERROR("[GDNLiveStateAllGatherStage] Workspace was not bound");
            return false;
        }
        if (!params_.conv_kernel || !params_.recurrence_kernel)
        {
            LOG_ERROR("[GDNLiveStateAllGatherStage] Both kernel state owners are required");
            return false;
        }

        void *stream = gpuStream();
        if (!stream)
        {
            LOG_ERROR("[GDNLiveStateAllGatherStage] Explicit non-null GPU stream is required");
            return false;
        }

        const auto conv_shape = params_.geometry.resolve(
            GDNLinkedLiveStateKind::ConvHistory,
            degree);
        const auto recurrence_shape = params_.geometry.resolve(
            GDNLinkedLiveStateKind::Recurrence,
            degree);
        if (!conv_shape || !recurrence_shape)
        {
            LOG_ERROR("[GDNLiveStateAllGatherStage] Invalid modulo-linked geometry"
                      << " key_heads=" << params_.geometry.global_key_heads
                      << " value_heads=" << params_.geometry.global_value_heads
                      << " key_width=" << params_.geometry.key_width
                      << " value_width=" << params_.geometry.value_width
                      << " history=" << params_.geometry.conv_history_length
                      << " degree=" << degree);
            return false;
        }

        const auto gather_state = [&]<typename Kernel>(
                                      Kernel *kernel,
                                      const GDNLinkedLiveStateShape &shape,
                                      const std::string &local_name,
                                      const std::string &gathered_name,
                                      const std::string &full_name,
                                      const char *collective_suffix) -> bool
        {
            auto *local = static_cast<float *>(
                bound_workspace_->getBuffer(local_name));
            auto *gathered = static_cast<float *>(
                bound_workspace_->getBuffer(gathered_name));
            auto *full = static_cast<float *>(
                bound_workspace_->getBuffer(full_name));
            if (!local || !gathered || !full)
            {
                LOG_ERROR("[GDNLiveStateAllGatherStage] Missing persistent "
                          << collective_suffix << " workspace buffers");
                return false;
            }

            // Export selects the participant-local bank by exact size. No
            // mutable host-side active-bank shadow participates in authority.
            if (!kernel->exportStateForSize(
                    shape.local_state_floats,
                    nullptr,
                    local,
                    stream))
            {
                LOG_ERROR("[GDNLiveStateAllGatherStage] Failed to export "
                          << collective_suffix << " state");
                return false;
            }
            if (!params_.tp_ctx->allgatherRawOnStream(
                    local,
                    gathered,
                    static_cast<size_t>(shape.local_state_floats),
                    CollectiveDataType::FLOAT32,
                    params_.tp_device_idx,
                    stream,
                    params_.stage_name + "_" + collective_suffix))
            {
                LOG_ERROR("[GDNLiveStateAllGatherStage] "
                          << collective_suffix << " state allgather failed");
                return false;
            }

            // A distinct destination avoids read/write overlap while each
            // thread maps one global group-major element from rank-major input.
            if (!reassembleOnDevice(gathered, full, shape, stream))
            {
                LOG_ERROR("[GDNLiveStateAllGatherStage] "
                          << collective_suffix
                          << " modulo-linked reassembly failed");
                return false;
            }
            if (!kernel->importStateForSize(
                    shape.full_state_floats,
                    nullptr,
                    full,
                    stream))
            {
                LOG_ERROR("[GDNLiveStateAllGatherStage] Failed to import full "
                          << collective_suffix << " state");
                return false;
            }
            return true;
        };

        const bool ok = gather_state(
                            params_.conv_kernel,
                            *conv_shape,
                            localConvBufferName(),
                            gatheredConvBufferName(),
                            fullConvBufferName(),
                            "conv") &&
                        gather_state(
                            params_.recurrence_kernel,
                            *recurrence_shape,
                            localRecurrenceBufferName(),
                            gatheredRecurrenceBufferName(),
                            fullRecurrenceBufferName(),
                            "recurrence");
        if (ok)
        {
            LOG_DEBUG("[GDNLiveStateAllGatherStage] Published modulo-linked full GDN state"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.toString()
                      << " tp_device_idx=" << params_.tp_device_idx
                      << " context=" << (context ? context : "unknown"));
        }
        return ok;
    }

    size_t GDNLiveStateAllGatherStage::estimatedMemoryBytes() const
    {
        if (!params_.tp_ctx)
            return 0;
        const auto conv = params_.geometry.resolve(
            GDNLinkedLiveStateKind::ConvHistory,
            params_.tp_ctx->degree());
        const auto recurrence = params_.geometry.resolve(
            GDNLinkedLiveStateKind::Recurrence,
            params_.tp_ctx->degree());
        if (!conv || !recurrence)
            return 0;
        const size_t total_floats =
            static_cast<size_t>(conv->local_state_floats) +
            2ULL * static_cast<size_t>(conv->full_state_floats) +
            static_cast<size_t>(recurrence->local_state_floats) +
            2ULL * static_cast<size_t>(recurrence->full_state_floats);
        return total_floats * sizeof(float);
    }

    bool GDNLiveStateAllGatherStage::supportsBackend(
        ComputeBackendType backend) const
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
        if (!params_.device_id.is_gpu() || !params_.tp_ctx ||
            !params_.conv_kernel || !params_.recurrence_kernel)
        {
            return false;
        }
        bool backend_supported = false;
#ifdef HAVE_CUDA
        backend_supported = backend_supported || params_.device_id.is_cuda();
#endif
#ifdef HAVE_ROCM
        backend_supported = backend_supported || params_.device_id.is_rocm();
#endif
        const int degree = params_.tp_ctx->degree();
        return backend_supported && degree > 1 &&
               params_.tp_device_idx >= 0 &&
               params_.tp_device_idx < degree &&
               params_.geometry.resolve(
                   GDNLinkedLiveStateKind::ConvHistory,
                   degree).has_value() &&
               params_.geometry.resolve(
                   GDNLinkedLiveStateKind::Recurrence,
                   degree).has_value() &&
               params_.tp_ctx->supportsRawAllgatherOnStreamGraphCapture();
    }

    StageDumpInfo GDNLiveStateAllGatherStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        info.addScalarInt("layer_idx", params_.layer_idx)
            .addScalarInt("tp_device_idx", params_.tp_device_idx)
            .addScalarInt("global_key_heads", params_.geometry.global_key_heads)
            .addScalarInt("global_value_heads", params_.geometry.global_value_heads)
            .addScalarInt("key_width", params_.geometry.key_width)
            .addScalarInt("value_width", params_.geometry.value_width)
            .addScalarInt(
                "conv_history_length",
                params_.geometry.conv_history_length);
        if (params_.tp_ctx)
        {
            if (const auto conv = params_.geometry.resolve(
                    GDNLinkedLiveStateKind::ConvHistory,
                    params_.tp_ctx->degree()))
            {
                info.addScalarInt(
                        "local_conv_state_floats",
                        conv->local_state_floats)
                    .addScalarInt(
                        "full_conv_state_floats",
                        conv->full_state_floats);
            }
            if (const auto recurrence = params_.geometry.resolve(
                    GDNLinkedLiveStateKind::Recurrence,
                    params_.tp_ctx->degree()))
            {
                info.addScalarInt(
                        "local_recurrence_state_floats",
                        recurrence->local_state_floats)
                    .addScalarInt(
                        "full_recurrence_state_floats",
                        recurrence->full_state_floats);
            }
        }
        return info;
    }

    StageBufferRequirements
    GDNLiveStateAllGatherStage::getBufferRequirements() const
    {
        return {};
    }

    StageBufferContract GDNLiveStateAllGatherStage::bufferContract() const
    {
        return {};
    }

    WorkspaceRequirements
    GDNLiveStateAllGatherStage::getWorkspaceRequirements(
        int m,
        int n,
        int k) const
    {
        (void)m;
        (void)n;
        (void)k;
        WorkspaceRequirements requirements;
        if (!params_.tp_ctx)
            return requirements;

        const auto append_state_buffers = [&requirements](
                                              const std::string &local,
                                              const std::string &gathered,
                                              const std::string &full,
                                              const GDNLinkedLiveStateShape &shape)
        {
            requirements.buffers.push_back(
                {local,
                 static_cast<size_t>(shape.local_state_floats) * sizeof(float),
                 256,
                 true});
            requirements.buffers.push_back(
                {gathered,
                 static_cast<size_t>(shape.full_state_floats) * sizeof(float),
                 256,
                 true});
            requirements.buffers.push_back(
                {full,
                 static_cast<size_t>(shape.full_state_floats) * sizeof(float),
                 256,
                 true});
        };

        if (const auto conv = params_.geometry.resolve(
                GDNLinkedLiveStateKind::ConvHistory,
                params_.tp_ctx->degree()))
        {
            append_state_buffers(
                localConvBufferName(),
                gatheredConvBufferName(),
                fullConvBufferName(),
                *conv);
        }
        if (const auto recurrence = params_.geometry.resolve(
                GDNLinkedLiveStateKind::Recurrence,
                params_.tp_ctx->degree()))
        {
            append_state_buffers(
                localRecurrenceBufferName(),
                gatheredRecurrenceBufferName(),
                fullRecurrenceBufferName(),
                *recurrence);
        }
        return requirements;
    }

    void GDNLiveStateAllGatherStage::bindWorkspace(
        DeviceWorkspaceManager *workspace)
    {
        bound_workspace_ = workspace;
    }

    void GDNLiveStateAllGatherStage::unbindWorkspace()
    {
        bound_workspace_ = nullptr;
    }
} // namespace llaminar2
