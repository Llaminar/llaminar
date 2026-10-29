/**
 * @file TPKVCacheStateAllGatherStage.cpp
 * @brief Exact live-row LocalTP K/V handoff for CPU and captured native GPUs.
 *
 * CPU projections are compacted into their admitted workspace and exchanged in
 * declared participant order. Separate per-participant full rows seed the same
 * canonical cache used by replicated decode. No unowned tensor capacity enters
 * the protocol, and the existing GPU stream/capture implementation is retained.
 */

#include "TPKVCacheStateAllGatherStage.h"
#include "../ComputeStageUtils.h"
#include "../../../collective/ILocalTPContext.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../memory/StageBufferContract.h"
#include "../../../tensors/ITensor.h"
#include "../../../tensors/TensorType.h"
#include "../../../utils/Logger.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

#ifdef HAVE_CUDA
extern "C"
{
    bool cudaTPKV_compact_rows_fp32(
        const float *src,
        float *dst,
        int tokens,
        int local_dim,
        int src_stride,
        int device_ordinal,
        void *stream);

    bool cudaTPKV_deinterleave_rank_major_fp32(
        const float *rank_major,
        float *row_major,
        int tokens,
        int degree,
        int local_dim,
        int device_ordinal,
        void *stream);
}
#endif

#ifdef HAVE_ROCM
extern "C"
{
    bool rocmTPKV_compact_rows_fp32(
        const float *src,
        float *dst,
        int tokens,
        int local_dim,
        int src_stride,
        int device_ordinal,
        void *stream);

    bool rocmTPKV_deinterleave_rank_major_fp32(
        const float *rank_major,
        float *row_major,
        int tokens,
        int degree,
        int local_dim,
        int device_ordinal,
        void *stream);
}
#endif

namespace llaminar2
{
    TPKVCacheStateAllGatherStage::TPKVCacheStateAllGatherStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params)),
          host_k_stage_name_((params_.stage_name.empty() ? "tp_kv_state_allgather" : params_.stage_name) + "_K"),
          host_v_stage_name_((params_.stage_name.empty() ? "tp_kv_state_allgather" : params_.stage_name) + "_V")
    {
    }

    std::string TPKVCacheStateAllGatherStage::gatheredKBufferName() const
    {
        return std::string(WS_GATHERED_K) + "_layer" + std::to_string(params_.layer_idx);
    }

    std::string TPKVCacheStateAllGatherStage::gatheredVBufferName() const
    {
        return std::string(WS_GATHERED_V) + "_layer" + std::to_string(params_.layer_idx);
    }

    std::string TPKVCacheStateAllGatherStage::compactKBufferName() const
    {
        return std::string(WS_COMPACT_K) + "_layer" + std::to_string(params_.layer_idx);
    }

    std::string TPKVCacheStateAllGatherStage::compactVBufferName() const
    {
        return std::string(WS_COMPACT_V) + "_layer" + std::to_string(params_.layer_idx);
    }

    bool TPKVCacheStateAllGatherStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "TPKVCacheStateAllGatherStage"))
            return false;

        if (params_.device_id.is_cpu())
        {
            if (ctx->deviceId() != params_.device_id)
                return false;
            return executeHost();
        }
        if (!params_.device_id.is_gpu())
        {
            LOG_ERROR("[TPKVCacheStateAllGatherStage] GPU device required, got "
                      << params_.device_id.toString());
            return false;
        }
        if (!params_.tp_ctx)
        {
            LOG_ERROR("[TPKVCacheStateAllGatherStage] Missing LocalTP context");
            return false;
        }
        if (params_.tp_device_idx < 0 || params_.tp_device_idx >= params_.tp_ctx->degree())
        {
            LOG_ERROR("[TPKVCacheStateAllGatherStage] Invalid TP device index "
                      << params_.tp_device_idx << " degree=" << params_.tp_ctx->degree());
            return false;
        }
        if (!bound_workspace_)
        {
            LOG_ERROR("[TPKVCacheStateAllGatherStage] Workspace was not bound");
            return false;
        }
        if (!params_.local_K || !params_.local_V || !params_.full_K || !params_.full_V)
        {
            LOG_ERROR("[TPKVCacheStateAllGatherStage] Missing K/V tensors");
            return false;
        }
        if (params_.local_K->native_type() != TensorType::FP32 ||
            params_.local_V->native_type() != TensorType::FP32 ||
            params_.full_K->native_type() != TensorType::FP32 ||
            params_.full_V->native_type() != TensorType::FP32)
        {
            LOG_ERROR("[TPKVCacheStateAllGatherStage] Only FP32 K/V tensors are supported"
                      << " local_K=" << tensorTypeName(params_.local_K->native_type())
                      << " local_V=" << tensorTypeName(params_.local_V->native_type())
                      << " full_K=" << tensorTypeName(params_.full_K->native_type())
                      << " full_V=" << tensorTypeName(params_.full_V->native_type()));
            return false;
        }

        const int degree = params_.tp_ctx->degree();
        if (params_.tokens <= 0 ||
            params_.local_kv_dim <= 0 ||
            params_.full_kv_dim <= 0 ||
            params_.local_k_stride < params_.local_kv_dim ||
            params_.local_v_stride < params_.local_kv_dim ||
            degree <= 1 ||
            params_.full_kv_dim != params_.local_kv_dim * degree)
        {
            LOG_ERROR("[TPKVCacheStateAllGatherStage] Invalid dimensions"
                      << " tokens=" << params_.tokens
                      << " local_kv_dim=" << params_.local_kv_dim
                      << " full_kv_dim=" << params_.full_kv_dim
                      << " local_k_stride=" << params_.local_k_stride
                      << " local_v_stride=" << params_.local_v_stride
                      << " degree=" << degree);
            return false;
        }
        if (params_.local_K->rows() < static_cast<size_t>(params_.tokens) ||
            params_.local_V->rows() < static_cast<size_t>(params_.tokens) ||
            params_.full_K->rows() < static_cast<size_t>(params_.tokens) ||
            params_.full_V->rows() < static_cast<size_t>(params_.tokens) ||
            params_.local_K->cols() < static_cast<size_t>(params_.local_kv_dim) ||
            params_.local_V->cols() < static_cast<size_t>(params_.local_kv_dim) ||
            params_.full_K->cols() < static_cast<size_t>(params_.full_kv_dim) ||
            params_.full_V->cols() < static_cast<size_t>(params_.full_kv_dim))
        {
            LOG_ERROR("[TPKVCacheStateAllGatherStage] Tensor shapes do not cover requested dimensions"
                      << " tokens=" << params_.tokens
                      << " local_kv_dim=" << params_.local_kv_dim
                      << " full_kv_dim=" << params_.full_kv_dim);
            return false;
        }
        const size_t local_k_required =
            static_cast<size_t>(params_.tokens - 1) * static_cast<size_t>(params_.local_k_stride) +
            static_cast<size_t>(params_.local_kv_dim);
        const size_t local_v_required =
            static_cast<size_t>(params_.tokens - 1) * static_cast<size_t>(params_.local_v_stride) +
            static_cast<size_t>(params_.local_kv_dim);
        if (params_.local_K->numel() < local_k_required ||
            params_.local_V->numel() < local_v_required)
        {
            LOG_ERROR("[TPKVCacheStateAllGatherStage] Explicit source strides exceed local K/V storage"
                      << " tokens=" << params_.tokens
                      << " local_kv_dim=" << params_.local_kv_dim
                      << " local_k_stride=" << params_.local_k_stride
                      << " local_v_stride=" << params_.local_v_stride
                      << " local_K_numel=" << params_.local_K->numel()
                      << " local_V_numel=" << params_.local_V->numel());
            return false;
        }

        void *stream = gpuStream();
        if (!stream)
        {
            LOG_ERROR("[TPKVCacheStateAllGatherStage] Explicit non-null GPU stream is required");
            return false;
        }

        auto *gathered_K = static_cast<float *>(bound_workspace_->getBuffer(gatheredKBufferName()));
        auto *gathered_V = static_cast<float *>(bound_workspace_->getBuffer(gatheredVBufferName()));
        auto *compact_K = static_cast<float *>(bound_workspace_->getBuffer(compactKBufferName()));
        auto *compact_V = static_cast<float *>(bound_workspace_->getBuffer(compactVBufferName()));
        if (!gathered_K || !gathered_V || !compact_K || !compact_V)
        {
            LOG_ERROR("[TPKVCacheStateAllGatherStage] Missing compact/gathered K/V workspace buffers");
            return false;
        }

        const int local_k_stride = params_.local_k_stride;
        const int local_v_stride = params_.local_v_stride;
        auto *local_K = static_cast<const float *>(params_.local_K->active_data_ptr());
        auto *local_V = static_cast<const float *>(params_.local_V->active_data_ptr());
        if (!local_K || !local_V)
        {
            LOG_ERROR("[TPKVCacheStateAllGatherStage] Missing local K/V input pointers");
            return false;
        }

        bool compact_ok = false;
#ifdef HAVE_CUDA
        if (params_.device_id.is_cuda())
        {
            compact_ok = cudaTPKV_compact_rows_fp32(
                local_K,
                compact_K,
                params_.tokens,
                params_.local_kv_dim,
                local_k_stride,
                params_.device_id.cuda_ordinal(),
                stream);
        }
#endif
#ifdef HAVE_ROCM
        if (params_.device_id.is_rocm())
        {
            compact_ok = rocmTPKV_compact_rows_fp32(
                local_K,
                compact_K,
                params_.tokens,
                params_.local_kv_dim,
                local_k_stride,
                params_.device_id.rocm_ordinal(),
                stream);
        }
#endif
        if (!compact_ok)
        {
            LOG_ERROR("[TPKVCacheStateAllGatherStage] K row compaction failed");
            return false;
        }

        compact_ok = false;
#ifdef HAVE_CUDA
        if (params_.device_id.is_cuda())
        {
            compact_ok = cudaTPKV_compact_rows_fp32(
                local_V,
                compact_V,
                params_.tokens,
                params_.local_kv_dim,
                local_v_stride,
                params_.device_id.cuda_ordinal(),
                stream);
        }
#endif
#ifdef HAVE_ROCM
        if (params_.device_id.is_rocm())
        {
            compact_ok = rocmTPKV_compact_rows_fp32(
                local_V,
                compact_V,
                params_.tokens,
                params_.local_kv_dim,
                local_v_stride,
                params_.device_id.rocm_ordinal(),
                stream);
        }
#endif
        if (!compact_ok)
        {
            LOG_ERROR("[TPKVCacheStateAllGatherStage] V row compaction failed");
            return false;
        }

        const size_t send_count =
            static_cast<size_t>(params_.tokens) * static_cast<size_t>(params_.local_kv_dim);
        const auto stage_name = params_.stage_name.empty()
                                    ? std::string("tp_kv_state_allgather")
                                    : params_.stage_name;

        if (!params_.tp_ctx->allgatherRawOnStream(
                compact_K,
                gathered_K,
                send_count,
                CollectiveDataType::FLOAT32,
                params_.tp_device_idx,
                stream,
                stage_name + "_K"))
        {
            LOG_ERROR("[TPKVCacheStateAllGatherStage] K allgather failed");
            return false;
        }
        if (!params_.tp_ctx->allgatherRawOnStream(
                compact_V,
                gathered_V,
                send_count,
                CollectiveDataType::FLOAT32,
                params_.tp_device_idx,
                stream,
                stage_name + "_V"))
        {
            LOG_ERROR("[TPKVCacheStateAllGatherStage] V allgather failed");
            return false;
        }

        auto *full_K = static_cast<float *>(params_.full_K->active_mutable_data_ptr());
        auto *full_V = static_cast<float *>(params_.full_V->active_mutable_data_ptr());
        if (!full_K || !full_V)
        {
            LOG_ERROR("[TPKVCacheStateAllGatherStage] Missing full K/V output pointers");
            return false;
        }

        compact_ok = false;
#ifdef HAVE_CUDA
        if (params_.device_id.is_cuda())
        {
            compact_ok = cudaTPKV_deinterleave_rank_major_fp32(
                gathered_K,
                full_K,
                params_.tokens,
                degree,
                params_.local_kv_dim,
                params_.device_id.cuda_ordinal(),
                stream);
        }
#endif
#ifdef HAVE_ROCM
        if (params_.device_id.is_rocm())
        {
            compact_ok = rocmTPKV_deinterleave_rank_major_fp32(
                gathered_K,
                full_K,
                params_.tokens,
                degree,
                params_.local_kv_dim,
                params_.device_id.rocm_ordinal(),
                stream);
        }
#endif
        if (!compact_ok)
        {
            LOG_ERROR("[TPKVCacheStateAllGatherStage] K deinterleave failed");
            return false;
        }

        compact_ok = false;
#ifdef HAVE_CUDA
        if (params_.device_id.is_cuda())
        {
            compact_ok = cudaTPKV_deinterleave_rank_major_fp32(
                gathered_V,
                full_V,
                params_.tokens,
                degree,
                params_.local_kv_dim,
                params_.device_id.cuda_ordinal(),
                stream);
        }
#endif
#ifdef HAVE_ROCM
        if (params_.device_id.is_rocm())
        {
            compact_ok = rocmTPKV_deinterleave_rank_major_fp32(
                gathered_V,
                full_V,
                params_.tokens,
                degree,
                params_.local_kv_dim,
                params_.device_id.rocm_ordinal(),
                stream);
        }
#endif
        if (!compact_ok)
        {
            LOG_ERROR("[TPKVCacheStateAllGatherStage] V deinterleave failed");
            return false;
        }

        LOG_DEBUG("[TPKVCacheStateAllGatherStage] Gathered full K/V state"
                  << " layer=" << params_.layer_idx
                  << " tokens=" << params_.tokens
                  << " local_kv_dim=" << params_.local_kv_dim
                  << " full_kv_dim=" << params_.full_kv_dim
                  << " degree=" << degree
                  << " device=" << params_.device_id.toString()
                  << " tp_device_idx=" << params_.tp_device_idx);
        return true;
    }

    bool TPKVCacheStateAllGatherStage::executeHost()
    {
        const auto reject = [&](const char *reason) {
            LOG_ERROR("[TPKVCacheStateAllGatherStage] " << reason);
            if (params_.tp_ctx) params_.tp_ctx->requestAbort();
            return false;
        };
        if (!params_.tp_ctx || params_.tp_ctx->backend() != CollectiveBackendType::HOST ||
            params_.tp_device_idx < 0 || params_.tp_device_idx >= params_.tp_ctx->degree() ||
            params_.tokens < 0 || params_.local_kv_dim <= 0 || params_.full_kv_dim <= 0 ||
            params_.local_k_stride < params_.local_kv_dim || params_.local_v_stride < params_.local_kv_dim ||
            static_cast<size_t>(params_.full_kv_dim) !=
                static_cast<size_t>(params_.local_kv_dim) * static_cast<size_t>(params_.tp_ctx->degree()) ||
            !params_.local_K || !params_.local_V || !params_.full_K || !params_.full_V)
        {
            return reject("Invalid indexed CPU handoff geometry");
        }
        const auto cpu_tensor = [](const ITensor *tensor) {
            // current_device() names a GPU allocation; a native host owner has
            // no such device. Require the existing host coherence authority so
            // active pointer access cannot choose a device buffer or transfer.
            const auto *native = dynamic_cast<const TensorBase *>(tensor);
            const auto *owner = native ? native->transferStorageOwner() : nullptr;
            return owner && tensor->native_type() == TensorType::FP32 &&
                owner->home_device().is_cpu() && !owner->current_device() && owner->hostValid();
        };
        if (!cpu_tensor(params_.local_K) || !cpu_tensor(params_.local_V) ||
            !cpu_tensor(params_.full_K) || !cpu_tensor(params_.full_V))
        {
            return reject("CPU handoff requires native CPU FP32 owners");
        }
        if (params_.tokens == 0)
            return params_.tp_ctx->allgatherHostIndexed({}, {}, params_.tp_device_idx, host_k_stage_name_) &&
                params_.tp_ctx->allgatherHostIndexed({}, {}, params_.tp_device_idx, host_v_stage_name_);
        const size_t rows = static_cast<size_t>(params_.tokens);
        const size_t width = static_cast<size_t>(params_.local_kv_dim);
        const size_t full_width = static_cast<size_t>(params_.full_kv_dim);
        if (rows > std::numeric_limits<size_t>::max() / (full_width * sizeof(float)) ||
            !bound_workspace_ || params_.local_K->rows() < rows || params_.local_V->rows() < rows ||
            params_.full_K->rows() < rows || params_.full_V->rows() < rows ||
            params_.local_K->cols() < width || params_.local_V->cols() < width ||
            params_.full_K->shape().size() != 2 || params_.full_V->shape().size() != 2 ||
            params_.full_K->cols() != full_width || params_.full_V->cols() != full_width ||
            params_.local_K->numel() < (rows - 1) * static_cast<size_t>(params_.local_k_stride) + width ||
            params_.local_V->numel() < (rows - 1) * static_cast<size_t>(params_.local_v_stride) + width)
        {
            return reject("CPU live rows exceed admitted owners/workspace");
        }
        if (bound_workspace_->device() != params_.device_id ||
            bound_workspace_->getBufferSize(compactKBufferName()) < rows * width * sizeof(float) ||
            bound_workspace_->getBufferSize(compactVBufferName()) < rows * width * sizeof(float) ||
            bound_workspace_->getBufferSize(gatheredKBufferName()) < rows * full_width * sizeof(float) ||
            bound_workspace_->getBufferSize(gatheredVBufferName()) < rows * full_width * sizeof(float))
        {
            return reject("CPU workspace owner or exact live extent is insufficient");
        }
        auto *compact_k = static_cast<float *>(bound_workspace_->getBuffer(compactKBufferName()));
        auto *compact_v = static_cast<float *>(bound_workspace_->getBuffer(compactVBufferName()));
        auto *gathered_k = static_cast<float *>(bound_workspace_->getBuffer(gatheredKBufferName()));
        auto *gathered_v = static_cast<float *>(bound_workspace_->getBuffer(gatheredVBufferName()));
        const auto *local_k = static_cast<const float *>(params_.local_K->active_data_ptr());
        const auto *local_v = static_cast<const float *>(params_.local_V->active_data_ptr());
        auto *full_k = static_cast<float *>(params_.full_K->active_mutable_data_ptr());
        auto *full_v = static_cast<float *>(params_.full_V->active_mutable_data_ptr());
        if (!compact_k || !compact_v || !gathered_k || !gathered_v || !local_k || !local_v || !full_k || !full_v)
        {
            return reject("Missing admitted CPU handoff pointers");
        }
        for (size_t row = 0; row < rows; ++row)
        {
            std::memcpy(compact_k + row * width, local_k + row * params_.local_k_stride, width * sizeof(float));
            std::memcpy(compact_v + row * width, local_v + row * params_.local_v_stride, width * sizeof(float));
        }
        const size_t local_elements = rows * width;
        const size_t full_elements = rows * full_width;
        if (!params_.tp_ctx->allgatherHostIndexed(std::as_bytes(std::span(compact_k, local_elements)),
                std::as_writable_bytes(std::span(gathered_k, full_elements)), params_.tp_device_idx, host_k_stage_name_) ||
            !params_.tp_ctx->allgatherHostIndexed(std::as_bytes(std::span(compact_v, local_elements)),
                std::as_writable_bytes(std::span(gathered_v, full_elements)), params_.tp_device_idx, host_v_stage_name_))
            return false;
        for (size_t row = 0; row < rows; ++row)
            for (size_t participant = 0; participant < static_cast<size_t>(params_.tp_ctx->degree()); ++participant)
            {
                const size_t source = participant * local_elements + row * width;
                const size_t destination = row * full_width + participant * width;
                std::memcpy(full_k + destination, gathered_k + source, width * sizeof(float));
                std::memcpy(full_v + destination, gathered_v + source, width * sizeof(float));
            }
        return true;
    }

    size_t TPKVCacheStateAllGatherStage::estimatedMemoryBytes() const
    {
        const int degree = params_.tp_ctx ? params_.tp_ctx->degree() : 0;
        if (params_.tokens <= 0 || params_.local_kv_dim <= 0 || degree <= 0)
            return 0;
        return 2ULL *
               static_cast<size_t>(params_.tokens) *
               static_cast<size_t>(params_.local_kv_dim) *
               static_cast<size_t>(degree + 1) *
               sizeof(float);
    }

    bool TPKVCacheStateAllGatherStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:
            return true;
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

    bool TPKVCacheStateAllGatherStage::isGraphCapturable() const
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
        if (params_.tokens <= 0 ||
            params_.local_kv_dim <= 0 ||
            params_.full_kv_dim <= 0 ||
            params_.local_k_stride < params_.local_kv_dim ||
            params_.local_v_stride < params_.local_kv_dim)
            return false;
        return params_.tp_ctx->supportsRawAllgatherOnStreamGraphCapture();
    }

    StageDumpInfo TPKVCacheStateAllGatherStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        info.addScalarInt("layer_idx", params_.layer_idx)
            .addScalarInt("tp_device_idx", params_.tp_device_idx)
            .addScalarInt("tokens", params_.tokens)
            .addScalarInt("local_kv_dim", params_.local_kv_dim)
            .addScalarInt("full_kv_dim", params_.full_kv_dim)
            .addScalarInt("local_k_stride", params_.local_k_stride)
            .addScalarInt("local_v_stride", params_.local_v_stride);
        if (params_.full_K)
            info.addOutput("full_K", params_.full_K, params_.full_K->rows(), params_.full_K->cols());
        if (params_.full_V)
            info.addOutput("full_V", params_.full_V, params_.full_V->rows(), params_.full_V->cols());
        if (params_.local_K)
            info.addInput("local_K", params_.local_K, params_.local_K->rows(), params_.local_K->cols());
        if (params_.local_V)
            info.addInput("local_V", params_.local_V, params_.local_V->rows(), params_.local_V->cols());
        return info;
    }

    StageBufferRequirements TPKVCacheStateAllGatherStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        if (params_.full_K)
            reqs.addOutput("full_K", params_.full_K->shape(), toBufferTensorType(params_.full_K->native_type()));
        if (params_.full_V)
            reqs.addOutput("full_V", params_.full_V->shape(), toBufferTensorType(params_.full_V->native_type()));
        return reqs;
    }

    StageBufferContract TPKVCacheStateAllGatherStage::bufferContract() const
    {
        if (!params_.local_k_buffer_id || !params_.local_v_buffer_id ||
            !params_.full_k_buffer_id || !params_.full_v_buffer_id)
        {
            return {};
        }

        return StageBufferContract::build()
            .addInput(*params_.local_k_buffer_id)
            .addInput(*params_.local_v_buffer_id)
            .addOutput(*params_.full_k_buffer_id)
            .addOutput(*params_.full_v_buffer_id);
    }

    WorkspaceRequirements TPKVCacheStateAllGatherStage::getWorkspaceRequirements(int m, int n, int k) const
    {
        (void)m;
        (void)n;
        (void)k;

        WorkspaceRequirements reqs;
        if (params_.tokens <= 0 || params_.local_kv_dim <= 0 || !params_.tp_ctx)
            return reqs;

        const size_t bytes =
            static_cast<size_t>(params_.tokens) *
            static_cast<size_t>(params_.local_kv_dim) *
            static_cast<size_t>(params_.tp_ctx->degree()) *
            sizeof(float);
        const size_t compact_bytes =
            static_cast<size_t>(params_.tokens) *
            static_cast<size_t>(params_.local_kv_dim) *
            sizeof(float);
        reqs.buffers.push_back({compactKBufferName(), compact_bytes, 256, true});
        reqs.buffers.push_back({compactVBufferName(), compact_bytes, 256, true});
        reqs.buffers.push_back({gatheredKBufferName(), bytes, 256, true});
        reqs.buffers.push_back({gatheredVBufferName(), bytes, 256, true});
        return reqs;
    }

    void TPKVCacheStateAllGatherStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        bound_workspace_ = workspace;
    }

    void TPKVCacheStateAllGatherStage::unbindWorkspace()
    {
        bound_workspace_ = nullptr;
    }

} // namespace llaminar2
