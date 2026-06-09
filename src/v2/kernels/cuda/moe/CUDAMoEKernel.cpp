/**
 * @file CUDAMoEKernel.cpp
 * @brief CUDA MoE kernel bridge implementation.
 *
 * The C++ bridge owns tensor coherence, persistent scratch allocation, and
 * error reporting while delegating actual CUDA launches to extern "C" wrappers
 * in `CUDAMoEKernels.cu`. This keeps MPI-heavy project headers out of nvcc
 * compilation and preserves the established CUDA backend split used by other
 * kernels.
 */

#include "CUDAMoEKernel.h"

#include "../gemm/CUDADeviceWorkspace.h"
#include "../../../execution/moe/MoERuntimeTable.h"
#include "../../../execution/moe/MoEWorkspaceRequirements.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../utils/Logger.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/PerfStatsCollector.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace
{
    bool setMoEDevice(int device_ordinal, const char *context)
    {
        cudaError_t err = cudaSetDevice(device_ordinal);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel] cudaSetDevice(" << device_ordinal
                                                        << ") failed in " << context << ": " << cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    llaminar2::TensorBase *asTensorBase(llaminar2::ITensor *tensor, const char *context)
    {
        auto *base = dynamic_cast<llaminar2::TensorBase *>(tensor);
        if (!base)
            LOG_ERROR("[CUDAMoEKernel] " << context << " requires TensorBase-backed tensor");
        return base;
    }

    bool ensureTensorOnDevice(llaminar2::ITensor *tensor, llaminar2::DeviceId device, void *stream, const char *name)
    {
        auto *base = asTensorBase(tensor, name);
        if (!base)
            return false;
        if (!base->gpu_data_ptr() && !base->allocateOnDevice(device, stream))
        {
            LOG_ERROR("[CUDAMoEKernel] Failed to allocate tensor '" << name << "' on " << device.to_string());
            return false;
        }
        if (!base->ensureOnDevice(device, stream))
        {
            LOG_ERROR("[CUDAMoEKernel] Failed to ensure tensor '" << name << "' on " << device.to_string());
            return false;
        }
        return base->gpu_data_ptr() != nullptr;
    }

    bool ensureOutputOnDevice(llaminar2::ITensor *tensor, llaminar2::DeviceId device, void *stream, const char *name)
    {
        auto *base = asTensorBase(tensor, name);
        if (!base)
            return false;
        if (!base->gpu_data_ptr() && !base->allocateOnDevice(device, stream))
        {
            LOG_ERROR("[CUDAMoEKernel] Failed to allocate output tensor '" << name << "' on " << device.to_string());
            return false;
        }
        return base->gpu_data_ptr() != nullptr;
    }

    void markDeviceWritten(llaminar2::ITensor *tensor, llaminar2::DeviceId device, void *stream)
    {
        if (auto *base = dynamic_cast<llaminar2::TensorBase *>(tensor))
            base->transitionToWithEvent(llaminar2::TensorCoherenceState::DEVICE_AUTHORITATIVE, device, stream);
        else
            tensor->transitionTo(llaminar2::TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    }

    void markSynced(llaminar2::ITensor *tensor)
    {
        tensor->transitionTo(llaminar2::TensorCoherenceState::SYNCED);
    }

    llaminar2::PerfStatsCollector::Tags groupedDecodeTags(
        const char *source,
        int active_slots,
        int d_model,
        int intermediate,
        const char *route)
    {
        return {
            {"source", source},
            {"active_slots", std::to_string(active_slots)},
            {"d_model", std::to_string(d_model)},
            {"intermediate", std::to_string(intermediate)},
            {"route", route}};
    }

    void recordGroupedDecodeCounter(
        const char *name,
        const char *source,
        int active_slots,
        int d_model,
        int intermediate,
        const char *route)
    {
        llaminar2::PerfStatsCollector::addCounter(
            "kernel", name, 1.0, {}, {},
            groupedDecodeTags(source, active_slots, d_model, intermediate, route));
    }

    void recordFusedDecodeTimer(
        const char *name,
        int top_k,
        int d_model,
        int intermediate)
    {
        llaminar2::PerfStatsCollector::recordTimingNs(
            "kernel_cuda", name, 1, {}, {},
            {{"source", "fused_runtime"},
             {"stage_type", "MOE_EXPERT_FFN"},
             {"top_k", std::to_string(top_k)},
             {"d_model", std::to_string(d_model)},
             {"intermediate", std::to_string(intermediate)}});
    }

    bool isAligned(const void *ptr, std::uintptr_t alignment)
    {
        return ptr && ((reinterpret_cast<std::uintptr_t>(ptr) & (alignment - 1)) == 0);
    }

    bool requireAlignedPointer(const void *ptr, std::uintptr_t alignment, const char *name, const char *context)
    {
        if (isAligned(ptr, alignment))
            return true;
        LOG_ERROR("[CUDAMoEKernel] " << context << " requires " << alignment
                                      << "-byte aligned " << name << " pointer, got " << ptr);
        return false;
    }

    bool requireTensorType(const llaminar2::TensorBase *tensor, llaminar2::TensorType expected,
                           const char *name, const char *context)
    {
        if (tensor && tensor->native_type() == expected)
            return true;
        LOG_ERROR("[CUDAMoEKernel] " << context << " requires " << name << " type "
                                      << llaminar2::tensorTypeName(expected)
                                      << ", got "
                                      << (tensor ? llaminar2::tensorTypeName(tensor->native_type()) : "null"));
        return false;
    }

    bool requireTensorTypeOneOf(const llaminar2::TensorBase *tensor,
                                llaminar2::TensorType a,
                                llaminar2::TensorType b,
                                const char *name,
                                const char *context)
    {
        if (tensor && (tensor->native_type() == a || tensor->native_type() == b))
            return true;
        LOG_ERROR("[CUDAMoEKernel] " << context << " requires " << name << " type "
                                      << llaminar2::tensorTypeName(a) << " or "
                                      << llaminar2::tensorTypeName(b)
                                      << ", got "
                                      << (tensor ? llaminar2::tensorTypeName(tensor->native_type()) : "null"));
        return false;
    }

    bool requireTensorElements(const llaminar2::ITensor *tensor, size_t required,
                               const char *name, const char *context)
    {
        if (tensor && tensor->numel() >= required)
            return true;
        LOG_ERROR("[CUDAMoEKernel] " << context << " requires " << name
                                      << " to have at least " << required
                                      << " elements, got " << (tensor ? tensor->numel() : 0));
        return false;
    }

    bool requireMatrixCapacity(const llaminar2::ITensor *tensor,
                               int required_rows,
                               int required_cols,
                               const char *name,
                               const char *context)
    {
        if (!tensor || required_rows < 0 || required_cols < 0)
            return false;
        const auto &shape = tensor->shape();
        const bool ok = shape.size() >= 2 &&
                        shape[0] >= static_cast<size_t>(required_rows) &&
                        shape[1] >= static_cast<size_t>(required_cols);
        if (ok)
            return true;
        LOG_ERROR("[CUDAMoEKernel] " << context << " requires " << name
                                      << " matrix capacity at least "
                                      << required_rows << "x" << required_cols
                                      << ", got shape "
                                      << (shape.empty() ? 0 : shape[0]) << "x"
                                      << (shape.size() > 1 ? shape[1] : 1));
        return false;
    }

    bool requireCudaDevicePointer(const void *ptr, int expected_device,
                                  const char *name, const char *context, void *stream)
    {
        if (!ptr)
        {
            LOG_ERROR("[CUDAMoEKernel] " << context << " requires non-null " << name << " pointer");
            return false;
        }
        if (llaminar2::isGraphCaptureActive())
            return true;
        if (stream)
        {
            cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
            const cudaError_t capture_err =
                cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &capture_status);
            if (capture_err == cudaSuccess && capture_status != cudaStreamCaptureStatusNone)
                return true;
            if (capture_err != cudaSuccess)
                cudaGetLastError();
        }

        cudaPointerAttributes attr{};
        const cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
        if (err != cudaSuccess)
        {
            cudaGetLastError();
            LOG_ERROR("[CUDAMoEKernel] " << context << " cudaPointerGetAttributes failed for "
                                          << name << " pointer " << ptr << ": "
                                          << cudaGetErrorString(err));
            return false;
        }
        const bool type_ok = attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged;
        if (type_ok && attr.device == expected_device)
            return true;
        LOG_ERROR("[CUDAMoEKernel] " << context << " requires " << name
                                      << " pointer on CUDA device " << expected_device
                                      << ", got attr.device=" << attr.device
                                      << " attr.type=" << static_cast<int>(attr.type)
                                      << " ptr=" << ptr);
        return false;
    }

    int selectGroupedPrefillTileM(int requested_tile_m, int max_tokens_per_expert)
    {
        switch (requested_tile_m)
        {
        case 2:
        case 4:
        case 8:
        case 16:
            return requested_tile_m;
        default:
            break;
        }

        if (max_tokens_per_expert <= 2)
            return 2;
        if (max_tokens_per_expert <= 4)
            return 4;
        if (max_tokens_per_expert <= 8)
            return 8;
        return 16;
    }

    llaminar2::PerfStatsCollector::Tags groupedPrefillTags(
        int seq_len,
        int top_k,
        int num_experts,
        int active_expert_slots,
        int tile_m)
    {
        const int total_slots = seq_len * top_k;
        const int tile_n = (tile_m <= 2) ? 64 : 128;
        return {
            {"seq_len", std::to_string(seq_len)},
            {"top_k", std::to_string(top_k)},
            {"total_slots", std::to_string(total_slots)},
            {"activation_quant_rows", std::to_string(seq_len)},
            {"active_expert_slots", std::to_string(active_expert_slots)},
            {"num_experts", std::to_string(num_experts)},
            {"tile_m", std::to_string(tile_m)},
            {"tile_n", std::to_string(tile_n)}};
    }

    void recordGroupedPrefillCounters(
        int seq_len,
        int top_k,
        int num_experts,
        int active_expert_slots,
        int tile_m,
        bool fuse_swiglu)
    {
        auto tags = groupedPrefillTags(seq_len, top_k, num_experts, active_expert_slots, tile_m);
        if (active_expert_slots > 0)
        {
            llaminar2::PerfStatsCollector::addCounter(
                "kernel", "cuda_moe_grouped_prefill_active_expert_grid_calls",
                1.0, {}, {}, tags);
        }

        tags["swiglu_path"] = fuse_swiglu ? "fused" : "split";
        if (active_expert_slots > 0)
        {
            tags["gateup_route"] = fuse_swiglu ? "kpart_swiglu" : "kpart_prefill";
            tags["down_route"] = "kpart_prefill";
            tags["down_accumulation"] = "token_direct";
        }
        else
        {
            tags["gateup_route"] = "serial";
            tags["down_route"] = "serial";
            tags["down_accumulation"] = "slot_scatter";
        }
        llaminar2::PerfStatsCollector::addCounter(
            "kernel", "cuda_moe_grouped_prefill_swiglu_path_calls",
            1.0, {}, {}, std::move(tags));
    }

    bool isCudaStreamCapturing(void *stream)
    {
        if (!stream)
            return false;
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        const cudaError_t err = cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &status);
        return err == cudaSuccess && status != cudaStreamCaptureStatusNone;
    }

    int *runtimeTopKExpertIdsDevice(llaminar2::DeviceMoELayerRuntime *runtime_layer)
    {
        auto *base = reinterpret_cast<char *>(runtime_layer);
        return reinterpret_cast<int *>(base + offsetof(llaminar2::DeviceMoELayerRuntime, topk_expert_ids));
    }

    float *runtimeTopKWeightsDevice(llaminar2::DeviceMoELayerRuntime *runtime_layer)
    {
        auto *base = reinterpret_cast<char *>(runtime_layer);
        return reinterpret_cast<float *>(base + offsetof(llaminar2::DeviceMoELayerRuntime, topk_weights));
    }

    uint64_t *runtimeDecodeHistogramDevice(llaminar2::DeviceMoELayerRuntime *runtime_layer)
    {
        auto *base = reinterpret_cast<char *>(runtime_layer);
        return reinterpret_cast<uint64_t *>(base + offsetof(llaminar2::DeviceMoELayerRuntime, decode_histogram));
    }

    bool cudaGroupedPrefillSupportsCodebook(uint8_t cb)
    {
        switch (cb)
        {
        case 0:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
        case 15:
        case 16:
        case 17:
        case 19:
            return true;
        default:
            return false;
        }
    }

    bool cudaGroupedPrefillRequiresMins(uint8_t cb)
    {
        switch (cb)
        {
        case 5:
        case 7:
        case 8:
        case 9:
        case 10:
        case 13:
        case 14:
        case 16:
        case 17:
            return true;
        default:
            return false;
        }
    }

    bool cudaGroupedPrefillRequiresEmins(uint8_t cb)
    {
        return cb == 10;
    }

    bool validateCudaGroupedDesc(
        const llaminar2::DeviceNativeVNNIMatrixDesc &desc,
        int n,
        int k,
        uint8_t codebook_id)
    {
        return desc.valid() &&
               desc.n == n &&
               desc.k == k &&
               desc.blocks_per_row == static_cast<uint32_t>(k / 32) &&
               desc.codebook_id == codebook_id &&
               (!cudaGroupedPrefillRequiresMins(codebook_id) || desc.mins != nullptr) &&
               (!cudaGroupedPrefillRequiresEmins(codebook_id) || desc.emins != nullptr);
    }
}

extern "C"
{
    bool cudaNativeVNNIInitIQGridTables_tuned();

    bool cudaMoE_route_logits(
        const float *hidden, const float *gate_weights, float *logits,
        int seq_len, int d_model, int num_experts,
        int device_idx, void *stream);

    bool cudaMoE_route_logits_bf16(
        const float *hidden, const void *gate_weights, float *logits,
        int seq_len, int d_model, int num_experts,
        int device_idx, void *stream);

    bool cudaMoE_softmax_topk(
        float *logits, int *expert_indices, float *expert_weights,
        int seq_len, int num_experts, int top_k, bool normalize_weights,
        int device_idx, void *stream);

    bool cudaMoE_softmax_topk_decode_runtime(
        float *logits, int *runtime_expert_ids, float *runtime_weights,
        uint64_t *runtime_histogram,
        float *legacy_indices, float *legacy_weights,
        int num_experts, int top_k, bool normalize_weights,
        bool write_legacy_outputs, bool update_runtime_histogram,
        int device_idx, void *stream);

    bool cudaMoE_decode_route_select_runtime(
        const int *expert_indices, const float *expert_weights,
        int *runtime_expert_ids, float *runtime_weights,
        uint64_t *runtime_histogram,
        float *legacy_indices, float *legacy_weights,
        int num_experts, int top_k, bool write_legacy_outputs,
        bool update_runtime_histogram, int device_idx, void *stream);

    bool cudaMoE_int_to_float(const int *input, float *output, int count, int device_idx, void *stream);
    bool cudaMoE_float_to_int(const float *input, int *output, int count, int device_idx, void *stream);

    bool cudaMoE_gather_tokens(
        const float *hidden, float *batch_buffer, const int *token_indices,
        int num_tokens, int d_model, int device_idx, void *stream);

    bool cudaMoE_scatter_add(
        float *output, const float *expert_output, const int *token_indices,
        const float *weights, int num_tokens, int d_model, int device_idx, void *stream);

    bool cudaMoE_shared_expert_gate(
        const float *input, const float *gate_inp, float *shared_output,
        int seq_len, int d_model, int device_idx, void *stream);

    bool cudaMoE_swiglu(float *gate, const float *up, int count, int device_idx, void *stream);
    bool cudaMoE_weighted_add(float *output, const float *input, float weight, int count, int device_idx, void *stream);

    bool cudaMoE_count_per_expert(
        const int *routing_indices, int *expert_counts, int total_slots,
        int num_experts, int device_idx, void *stream);

    bool cudaMoE_exclusive_scan(
        const int *expert_counts, int *expert_offsets,
        int num_experts, int device_idx, void *stream);

    bool cudaMoE_scatter_tokens(
        const int *routing_indices, const float *routing_weights,
        int *write_heads, const int *expert_offsets,
        int *grouped_token_indices, float *grouped_weights,
        int total_slots, int top_k, int num_experts,
        int device_idx, void *stream);

    bool cudaMoE_scatter_tokens_deterministic(
        const int *routing_indices,
        const float *routing_weights,
        const int *expert_offsets,
        const int *expert_counts,
        int *grouped_token_indices,
        int *original_to_grouped,
        int *original_expert_ids,
        float *grouped_weights,
        int total_slots,
        int top_k,
        int num_experts,
        int device_idx,
        void *stream);

    bool cudaMoE_group_tokens_small_float(
        const float *routing_indices,
        const float *routing_weights,
        int *expert_counts,
        int *expert_offsets,
        int *grouped_token_indices,
        int *original_to_grouped,
        int *original_expert_ids,
        float *grouped_weights,
        int *active_expert_ids,
        int total_slots,
        int num_experts,
        int top_k,
        int max_active_experts,
        int device_idx,
        void *stream);

    bool cudaMoE_prepare_shared_expert_group(
        int *expert_offsets,
        int *expert_counts,
        int *grouped_token_indices,
        float *grouped_weights,
        int *active_expert_ids,
        int seq_len,
        int device_idx,
        void *stream);

    bool cudaMoE_gather_expert_fixed(
        const float *hidden, float *batch_buffer,
        const int *expert_offsets, const int *expert_counts,
        const int *grouped_token_indices,
        int expert_id, int max_tokens, int d_model,
        int device_idx, void *stream);

    bool cudaMoE_scatter_expert_fixed(
        float *output, const float *expert_output,
        const int *expert_offsets, const int *expert_counts,
        const int *grouped_token_indices,
        const float *grouped_weights,
        int expert_id, int max_tokens, int d_model,
        int device_idx, void *stream);

    bool cudaMoE_grouped_gate_up_native_vnni_decode_table(
        const float *d_hidden,
        const llaminar2::DeviceNativeVNNIMatrixDesc *d_gate_desc_table,
        const llaminar2::DeviceNativeVNNIMatrixDesc *d_up_desc_table,
        const int *d_expert_ids,
        float *const *d_gate_outputs,
        float *const *d_up_outputs,
        int8_t *d_hidden_int8,
        float *d_hidden_scales,
        int num_active,
        int intermediate,
        int d_model,
        uint8_t codebook_id,
        int device_idx,
        void *stream);

    bool cudaMoE_grouped_gate_up_native_vnni_decode_table_kpart(
        const float *d_hidden,
        const llaminar2::DeviceNativeVNNIMatrixDesc *d_gate_desc_table,
        const llaminar2::DeviceNativeVNNIMatrixDesc *d_up_desc_table,
        const int *d_expert_ids,
        float *const *d_gate_outputs,
        float *const *d_up_outputs,
        int8_t *d_hidden_int8,
        float *d_hidden_scales,
        float *d_gate_partials,
        float *d_up_partials,
        int num_active,
        int intermediate,
        int d_model,
        int num_experts,
        uint8_t codebook_id,
        int k_partitions,
        int device_idx,
        void *stream);

    bool cudaMoE_grouped_swiglu_down_native_vnni_decode_table(
        const float *const *d_gate_ptrs,
        const float *const *d_up_ptrs,
        const llaminar2::DeviceNativeVNNIMatrixDesc *d_desc_table,
        const int *d_expert_ids,
        const float *d_weights,
        int8_t *d_swiglu_int8,
        float *d_swiglu_scales,
        float *d_output,
        int num_active,
        int d_model,
        int intermediate,
        uint8_t codebook_id,
        int device_idx,
        void *stream);

    bool cudaMoE_grouped_swiglu_down_native_vnni_decode_table_kpart(
        const float *const *d_gate_ptrs,
        const float *const *d_up_ptrs,
        const llaminar2::DeviceNativeVNNIMatrixDesc *d_desc_table,
        const int *d_expert_ids,
        const float *d_weights,
        int8_t *d_swiglu_int8,
        float *d_swiglu_scales,
        float *d_down_partials,
        float *d_output,
        int num_active,
        int d_model,
        int intermediate,
        int num_experts,
        uint8_t codebook_id,
        int k_partitions,
        int device_idx,
        void *stream);

    bool cudaMoE_grouped_prefill_pipeline(
        const float *d_hidden,
        const llaminar2::DeviceNativeVNNIMatrixDesc *d_gate_desc_table,
        const llaminar2::DeviceNativeVNNIMatrixDesc *d_up_desc_table,
        const llaminar2::DeviceNativeVNNIMatrixDesc *d_down_desc_table,
        const int *d_group_counts,
        const int *d_group_offsets,
        const int *d_group_token_indices,
        const int *d_original_to_grouped,
        const int *d_active_expert_ids,
        const float *d_group_weights,
        int8_t *d_scratch_A_int8,
        float *d_scratch_scales,
        float *d_scratch_gate,
        float *d_scratch_up,
        int8_t *d_scratch_swiglu_int8,
        float *d_scratch_swiglu_scales,
        float *d_scratch_down_out,
        float *d_output,
        int num_experts,
        int d_model,
        int intermediate,
        int max_tokens_per_expert,
        int total_slots,
        int top_k,
        int active_expert_slots,
        uint8_t gateup_codebook_id,
        uint8_t down_codebook_id,
        int device_idx,
        void *stream);
}

namespace llaminar2
{
    CUDAMoEKernel::CUDAMoEKernel(int device_ordinal)
        : device_ordinal_(device_ordinal)
    {
        setMoEDevice(device_ordinal_, "CUDAMoEKernel::CUDAMoEKernel");
    }

    CUDAMoEKernel::~CUDAMoEKernel()
    {
        releaseDeviceBuffers();
    }

    void CUDAMoEKernel::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        CUDAKernelBase::bindWorkspace(workspace);
        clearWorkspaceScratchBindings();
    }

    bool CUDAMoEKernel::bindWorkspaceBuffer(
        void **ptr,
        const char *name,
        size_t bytes,
        const char *context)
    {
        if (!ptr || !name || bytes == 0)
            return false;
        if (!workspace_)
        {
            LOG_ERROR("[CUDAMoEKernel] " << context
                                         << " requires graph-owned MoE workspace");
            return false;
        }
        void *buffer = workspace_->getBuffer(name);
        const size_t available = workspace_->getBufferSize(name);
        if (!buffer || available < bytes)
        {
            LOG_ERROR("[CUDAMoEKernel] " << context << " missing required workspace buffer '"
                                         << name << "' (need " << bytes << " bytes, have "
                                         << available << ")");
            return false;
        }
        *ptr = buffer;
        scratch_workspace_bound_ = true;
        return true;
    }

    void CUDAMoEKernel::clearWorkspaceScratchBindings() noexcept
    {
        d_staging_indices_ = nullptr;
        d_staging_weights_ = nullptr;
        staging_capacity_ = 0;
        d_route_logits_ = nullptr;
        d_route_indices_ = nullptr;
        d_route_weights_ = nullptr;
        route_logits_capacity_ = 0;
        route_topk_capacity_ = 0;
        route_buffers_workspace_bound_ = false;
        d_group_int_indices_ = nullptr;
        d_group_offsets_ = nullptr;
        d_group_counts_ = nullptr;
        d_group_token_indices_ = nullptr;
        d_group_original_to_grouped_ = nullptr;
        d_group_original_expert_ids_ = nullptr;
        d_group_write_heads_ = nullptr;
        d_group_weights_ = nullptr;
        d_group_active_expert_ids_ = nullptr;
        group_active_expert_slots_ = 0;
        group_slots_cap_ = 0;
        group_experts_cap_ = 0;
        d_prefill_A_int8_ = nullptr;
        d_prefill_A_scales_ = nullptr;
        d_prefill_swiglu_int8_ = nullptr;
        d_prefill_swiglu_scales_ = nullptr;
        d_prefill_gate_ = nullptr;
        d_prefill_up_ = nullptr;
        prefill_slots_cap_ = 0;
        prefill_d_model_cap_ = 0;
        prefill_intermediate_cap_ = 0;
        d_decode_hidden_int8_ = nullptr;
        d_decode_hidden_scales_ = nullptr;
        decode_gateup_topk_cap_ = 0;
        decode_gateup_d_model_cap_ = 0;
        d_grouped_gateup_gate_partials_ = nullptr;
        d_grouped_gateup_up_partials_ = nullptr;
        grouped_gateup_kpart_active_cap_ = 0;
        grouped_gateup_kpart_partitions_cap_ = 0;
        grouped_gateup_kpart_intermediate_cap_ = 0;
        d_grouped_down_partials_ = nullptr;
        grouped_down_kpart_partitions_cap_ = 0;
        grouped_down_kpart_d_model_cap_ = 0;
        grouped_down_kpart_slots_cap_ = 0;
        d_decode_swiglu_int8_ = nullptr;
        d_decode_swiglu_scales_ = nullptr;
        decode_down_topk_cap_ = 0;
        decode_down_intermediate_cap_ = 0;
        d_grouped_decode_expert_ids_ = nullptr;
        d_grouped_decode_weights_ = nullptr;
        grouped_decode_metadata_cap_ = 0;
        grouped_decode_cached_expert_ids_.clear();
        grouped_decode_cached_weights_.clear();
        scratch_workspace_bound_ = false;
    }

    void CUDAMoEKernel::resetDynamicState()
    {
        host_expert_counts_.clear();
        host_expert_offsets_.clear();
        host_grouped_indices_.clear();
        host_grouped_weights_.clear();
        prepared_num_experts_ = 0;
        group_active_expert_slots_ = 0;
    }

    void CUDAMoEKernel::releaseDeviceBuffers() noexcept
    {
        cudaSetDevice(device_ordinal_);
        auto release = [](auto *&ptr)
        {
            if (ptr)
            {
                cudaFree(ptr);
                ptr = nullptr;
            }
        };

        if (scratch_workspace_bound_)
        {
            clearWorkspaceScratchBindings();
        }
        else
        {
            release(d_staging_indices_);
            release(d_staging_weights_);
            release(d_route_logits_);
            release(d_route_indices_);
            release(d_route_weights_);
            release(d_group_int_indices_);
            release(d_group_offsets_);
            release(d_group_counts_);
            release(d_group_token_indices_);
            release(d_group_original_to_grouped_);
            release(d_group_original_expert_ids_);
            release(d_group_active_expert_ids_);
            release(d_group_write_heads_);
            release(d_group_weights_);
            release(d_prefill_A_int8_);
            release(d_prefill_A_scales_);
            release(d_prefill_swiglu_int8_);
            release(d_prefill_swiglu_scales_);
            release(d_prefill_gate_);
            release(d_prefill_up_);
            release(d_decode_hidden_int8_);
            release(d_decode_hidden_scales_);
            release(d_grouped_gateup_gate_partials_);
            release(d_grouped_gateup_up_partials_);
            release(d_grouped_down_partials_);
            release(d_decode_swiglu_int8_);
            release(d_decode_swiglu_scales_);
            release(d_grouped_decode_expert_ids_);
            release(d_grouped_decode_weights_);
        }
        for (auto &table : grouped_down_desc_tables_)
            release(table.device_descs);
        for (auto &table : grouped_gateup_desc_tables_)
        {
            release(table.device_gate_descs);
            release(table.device_up_descs);
        }
        for (auto &entry : runtime_gateup_pointer_cache_)
        {
            release(entry.d_gate_ptrs);
            release(entry.d_up_ptrs);
        }
        for (auto &entry : runtime_down_pointer_cache_)
        {
            release(entry.d_gate_ptrs);
            release(entry.d_up_ptrs);
        }

        staging_capacity_ = 0;
        route_logits_capacity_ = 0;
        route_topk_capacity_ = 0;
        group_slots_cap_ = 0;
        group_experts_cap_ = 0;
        group_active_expert_slots_ = 0;
        prefill_slots_cap_ = 0;
        prefill_d_model_cap_ = 0;
        prefill_intermediate_cap_ = 0;
        decode_gateup_topk_cap_ = 0;
        decode_gateup_d_model_cap_ = 0;
        grouped_gateup_kpart_active_cap_ = 0;
        grouped_gateup_kpart_partitions_cap_ = 0;
        grouped_gateup_kpart_intermediate_cap_ = 0;
        decode_down_topk_cap_ = 0;
        decode_down_intermediate_cap_ = 0;
        grouped_decode_metadata_cap_ = 0;
        grouped_decode_cached_expert_ids_.clear();
        grouped_decode_cached_weights_.clear();
        grouped_down_kpart_partitions_cap_ = 0;
        grouped_down_kpart_d_model_cap_ = 0;
        grouped_down_kpart_slots_cap_ = 0;
        grouped_down_desc_tables_.clear();
        grouped_gateup_desc_tables_.clear();
        runtime_gateup_pointer_cache_.clear();
        runtime_down_pointer_cache_.clear();

        if (router_cublas_handle_)
        {
            cublasDestroy(static_cast<cublasHandle_t>(router_cublas_handle_));
            router_cublas_handle_ = nullptr;
        }
    }

    bool CUDAMoEKernel::routeLogitsCuBLAS(const float *hidden, const float *gate_weights, float *logits,
                                          int seq_len, int d_model, int num_experts)
    {
        void *stream = getStream();
        if (!stream)
        {
            LOG_ERROR("[CUDAMoEKernel::routeLogitsCuBLAS] CUDA router requires an explicit stream");
            return false;
        }

        if (!router_cublas_handle_)
        {
            if (isGraphCaptureActive() || isCudaStreamCapturing(stream))
            {
                LOG_ERROR("[CUDAMoEKernel::routeLogitsCuBLAS] cuBLAS handle was not warmed before graph capture");
                return false;
            }
            if (!setMoEDevice(device_ordinal_, "CUDAMoEKernel::routeLogitsCuBLAS"))
                return false;
            cublasHandle_t handle = nullptr;
            const cublasStatus_t create_status = cublasCreate(&handle);
            if (create_status != CUBLAS_STATUS_SUCCESS)
            {
                LOG_ERROR("[CUDAMoEKernel::routeLogitsCuBLAS] cublasCreate failed: "
                          << static_cast<int>(create_status));
                return false;
            }
            cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
            router_cublas_handle_ = handle;
        }

        auto *handle = static_cast<cublasHandle_t>(router_cublas_handle_);
        cublasStatus_t status = cublasSetStream(handle, static_cast<cudaStream_t>(stream));
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            LOG_ERROR("[CUDAMoEKernel::routeLogitsCuBLAS] cublasSetStream failed: "
                      << static_cast<int>(status));
            return false;
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;
        // Row-major logits[S,E] = hidden[S,D] * gate[E,D]^T.
        // In cuBLAS column-major views this is C^T[E,S] = gate[E,D] * hidden^T[D,S].
        status = cublasSgemm(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            num_experts, seq_len, d_model,
            &alpha,
            gate_weights, d_model,
            hidden, d_model,
            &beta,
            logits, num_experts);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            LOG_ERROR("[CUDAMoEKernel::routeLogitsCuBLAS] cublasSgemm failed: "
                      << static_cast<int>(status)
                      << " seq_len=" << seq_len
                      << " d_model=" << d_model
                      << " num_experts=" << num_experts);
            return false;
        }
        return true;
    }

    bool CUDAMoEKernel::ensureStagingCapacity(int count)
    {
        if (count <= staging_capacity_)
            return true;

        void *staging_indices = nullptr;
        void *staging_weights = nullptr;
        const bool ok =
            bindWorkspaceBuffer(&staging_indices, MoEWorkspaceBuffers::STAGING_INDICES,
                                static_cast<size_t>(count) * sizeof(int), "staging indices") &&
            bindWorkspaceBuffer(&staging_weights, MoEWorkspaceBuffers::STAGING_WEIGHTS,
                                static_cast<size_t>(count) * sizeof(float), "staging weights");
        if (!ok)
        {
            d_staging_indices_ = nullptr;
            d_staging_weights_ = nullptr;
            staging_capacity_ = 0;
            return false;
        }

        d_staging_indices_ = static_cast<int *>(staging_indices);
        d_staging_weights_ = static_cast<float *>(staging_weights);
        staging_capacity_ = count;
        return true;
    }

    bool CUDAMoEKernel::ensureRouteBufferCapacity(size_t logits_count, size_t topk_count)
    {
        if (logits_count <= route_logits_capacity_ && topk_count <= route_topk_capacity_)
            return true;

        void *route_logits = nullptr;
        void *route_indices = nullptr;
        void *route_weights = nullptr;
        const bool ok =
            bindWorkspaceBuffer(&route_logits, MoEWorkspaceBuffers::ROUTE_LOGITS,
                                logits_count * sizeof(float), "route logits") &&
            bindWorkspaceBuffer(&route_indices, MoEWorkspaceBuffers::ROUTE_INDICES,
                                topk_count * sizeof(int), "route indices") &&
            bindWorkspaceBuffer(&route_weights, MoEWorkspaceBuffers::ROUTE_WEIGHTS,
                                topk_count * sizeof(float), "route weights");
        if (!ok)
        {
            d_route_logits_ = nullptr;
            d_route_indices_ = nullptr;
            d_route_weights_ = nullptr;
            route_logits_capacity_ = 0;
            route_topk_capacity_ = 0;
            route_buffers_workspace_bound_ = false;
            return false;
        }

        d_route_logits_ = static_cast<float *>(route_logits);
        d_route_indices_ = static_cast<int *>(route_indices);
        d_route_weights_ = static_cast<float *>(route_weights);
        route_logits_capacity_ = logits_count;
        route_topk_capacity_ = topk_count;
        route_buffers_workspace_bound_ = true;
        return true;
    }

    bool CUDAMoEKernel::ensureGroupingBufferCapacity(int total_slots, int num_experts)
    {
        if (total_slots <= group_slots_cap_ && num_experts <= group_experts_cap_)
            return true;

        void *group_int_indices = nullptr;
        void *group_token_indices = nullptr;
        void *group_original_to_grouped = nullptr;
        void *group_original_expert_ids = nullptr;
        void *group_weights = nullptr;
        void *group_offsets = nullptr;
        void *group_counts = nullptr;
        void *group_active_expert_ids = nullptr;
        void *group_write_heads = nullptr;

        const bool ok =
            bindWorkspaceBuffer(&group_int_indices, MoEWorkspaceBuffers::GROUP_INT_INDICES,
                                static_cast<size_t>(total_slots) * sizeof(int), "group int indices") &&
            bindWorkspaceBuffer(&group_token_indices, MoEWorkspaceBuffers::GROUP_TOKEN_INDICES,
                                static_cast<size_t>(total_slots) * sizeof(int), "group token indices") &&
            bindWorkspaceBuffer(&group_original_to_grouped, MoEWorkspaceBuffers::GROUP_ORIGINAL_TO_GROUPED,
                                static_cast<size_t>(total_slots) * sizeof(int), "group original-to-grouped") &&
            bindWorkspaceBuffer(&group_original_expert_ids, MoEWorkspaceBuffers::GROUP_ORIGINAL_EXPERT_IDS,
                                static_cast<size_t>(total_slots) * sizeof(int), "group original expert ids") &&
            bindWorkspaceBuffer(&group_weights, MoEWorkspaceBuffers::GROUP_WEIGHTS,
                                static_cast<size_t>(total_slots) * sizeof(float), "group weights") &&
            bindWorkspaceBuffer(&group_offsets, MoEWorkspaceBuffers::GROUP_OFFSETS,
                                static_cast<size_t>(num_experts) * sizeof(int), "group offsets") &&
            bindWorkspaceBuffer(&group_counts, MoEWorkspaceBuffers::GROUP_COUNTS,
                                static_cast<size_t>(num_experts) * sizeof(int), "group counts") &&
            bindWorkspaceBuffer(&group_active_expert_ids, MoEWorkspaceBuffers::GROUP_ACTIVE_EXPERT_IDS,
                                static_cast<size_t>(num_experts) * sizeof(int), "group active expert ids") &&
            bindWorkspaceBuffer(&group_write_heads, MoEWorkspaceBuffers::GROUP_WRITE_HEADS,
                                static_cast<size_t>(num_experts) * sizeof(int), "group write heads");
        if (!ok)
        {
            d_group_int_indices_ = nullptr;
            d_group_token_indices_ = nullptr;
            d_group_original_to_grouped_ = nullptr;
            d_group_original_expert_ids_ = nullptr;
            d_group_weights_ = nullptr;
            d_group_offsets_ = nullptr;
            d_group_counts_ = nullptr;
            d_group_active_expert_ids_ = nullptr;
            d_group_write_heads_ = nullptr;
            group_slots_cap_ = 0;
            group_experts_cap_ = 0;
            return false;
        }

        d_group_int_indices_ = static_cast<int *>(group_int_indices);
        d_group_token_indices_ = static_cast<int *>(group_token_indices);
        d_group_original_to_grouped_ = static_cast<int *>(group_original_to_grouped);
        d_group_original_expert_ids_ = static_cast<int *>(group_original_expert_ids);
        d_group_weights_ = static_cast<float *>(group_weights);
        d_group_offsets_ = static_cast<int *>(group_offsets);
        d_group_counts_ = static_cast<int *>(group_counts);
        d_group_active_expert_ids_ = static_cast<int *>(group_active_expert_ids);
        d_group_write_heads_ = static_cast<int *>(group_write_heads);
        group_slots_cap_ = total_slots;
        group_experts_cap_ = num_experts;
        return true;
    }

    bool CUDAMoEKernel::ensureGroupedPrefillScratchCapacity(int total_slots, int d_model, int intermediate)
    {
        const bool need_realloc = total_slots > prefill_slots_cap_ ||
                                  d_model > prefill_d_model_cap_ ||
                                  intermediate > prefill_intermediate_cap_;
        if (!need_realloc)
            return true;

        const int max_dim = std::max(d_model, intermediate);
        const int max_blocks = (max_dim + 31) / 32;
        const int intermediate_blocks = (intermediate + 31) / 32;
        void *prefill_a_int8 = nullptr;
        void *prefill_a_scales = nullptr;
        void *prefill_swiglu_int8 = nullptr;
        void *prefill_swiglu_scales = nullptr;
        void *prefill_gate = nullptr;
        void *prefill_up = nullptr;
        const bool ok =
            bindWorkspaceBuffer(&prefill_a_int8, MoEWorkspaceBuffers::PREFILL_A_INT8,
                                static_cast<size_t>(total_slots) * max_dim * sizeof(int8_t), "prefill A int8") &&
            bindWorkspaceBuffer(&prefill_a_scales, MoEWorkspaceBuffers::PREFILL_A_SCALES,
                                static_cast<size_t>(total_slots) * max_blocks * sizeof(float), "prefill A scales") &&
            bindWorkspaceBuffer(&prefill_swiglu_int8, MoEWorkspaceBuffers::PREFILL_SWIGLU_INT8,
                                static_cast<size_t>(total_slots) * intermediate * sizeof(int8_t), "prefill SwiGLU int8") &&
            bindWorkspaceBuffer(&prefill_swiglu_scales, MoEWorkspaceBuffers::PREFILL_SWIGLU_SCALES,
                                static_cast<size_t>(total_slots) * intermediate_blocks * sizeof(float), "prefill SwiGLU scales") &&
            bindWorkspaceBuffer(&prefill_gate, MoEWorkspaceBuffers::PREFILL_GATE,
                                static_cast<size_t>(total_slots) * max_dim * sizeof(float), "prefill gate") &&
            bindWorkspaceBuffer(&prefill_up, MoEWorkspaceBuffers::PREFILL_UP,
                                static_cast<size_t>(total_slots) * intermediate * sizeof(float), "prefill up");
        if (!ok)
        {
            d_prefill_A_int8_ = nullptr;
            d_prefill_A_scales_ = nullptr;
            d_prefill_swiglu_int8_ = nullptr;
            d_prefill_swiglu_scales_ = nullptr;
            d_prefill_gate_ = nullptr;
            d_prefill_up_ = nullptr;
            prefill_slots_cap_ = 0;
            prefill_d_model_cap_ = 0;
            prefill_intermediate_cap_ = 0;
            return false;
        }

        d_prefill_A_int8_ = static_cast<int8_t *>(prefill_a_int8);
        d_prefill_A_scales_ = static_cast<float *>(prefill_a_scales);
        d_prefill_swiglu_int8_ = static_cast<int8_t *>(prefill_swiglu_int8);
        d_prefill_swiglu_scales_ = static_cast<float *>(prefill_swiglu_scales);
        d_prefill_gate_ = static_cast<float *>(prefill_gate);
        d_prefill_up_ = static_cast<float *>(prefill_up);
        prefill_slots_cap_ = total_slots;
        prefill_d_model_cap_ = d_model;
        prefill_intermediate_cap_ = intermediate;
        return true;
    }

    bool CUDAMoEKernel::ensureGroupedGateUpDecodeCapacity(int top_k, int d_model)
    {
        if (top_k <= 0 || top_k > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            d_model <= 0 || (d_model % 32) != 0)
            return false;

        const bool need_growth = top_k > decode_gateup_topk_cap_ ||
                                 d_model > decode_gateup_d_model_cap_ ||
                                 !d_decode_hidden_int8_ || !d_decode_hidden_scales_;
        if (!need_growth)
            return true;

        const int blocks_per_row = d_model / 32;
        void *decode_hidden_int8 = nullptr;
        void *decode_hidden_scales = nullptr;
        const bool ok =
            bindWorkspaceBuffer(&decode_hidden_int8, MoEWorkspaceBuffers::DECODE_HIDDEN_INT8,
                                static_cast<size_t>(d_model) * sizeof(int8_t), "decode hidden int8") &&
            bindWorkspaceBuffer(&decode_hidden_scales, MoEWorkspaceBuffers::DECODE_HIDDEN_SCALES,
                                static_cast<size_t>(blocks_per_row) * sizeof(float), "decode hidden scales");
        if (!ok)
        {
            d_decode_hidden_int8_ = nullptr;
            d_decode_hidden_scales_ = nullptr;
            decode_gateup_topk_cap_ = 0;
            decode_gateup_d_model_cap_ = 0;
            return false;
        }

        d_decode_hidden_int8_ = static_cast<int8_t *>(decode_hidden_int8);
        d_decode_hidden_scales_ = static_cast<float *>(decode_hidden_scales);
        decode_gateup_topk_cap_ = top_k;
        decode_gateup_d_model_cap_ = d_model;
        return true;
    }

    bool CUDAMoEKernel::ensureGroupedGateUpKPartScratchCapacity(int top_k, int k_partitions, int intermediate)
    {
        // Only the discrete partition counts the kpart launcher accepts are valid.
        if (top_k <= 0 || intermediate <= 0 ||
            !(k_partitions == 2 || k_partitions == 4 || k_partitions == 8 ||
              k_partitions == 16 || k_partitions == 32))
            return false;

        // Fast path: existing buffers already cover the requested shape.
        if (d_grouped_gateup_gate_partials_ && d_grouped_gateup_up_partials_ &&
            grouped_gateup_kpart_active_cap_ >= top_k &&
            grouped_gateup_kpart_partitions_cap_ >= k_partitions &&
            grouped_gateup_kpart_intermediate_cap_ >= intermediate)
            return true;

        const size_t partial_count = static_cast<size_t>(top_k) *
                                     static_cast<size_t>(k_partitions) *
                                     static_cast<size_t>(intermediate);
        void *gate_partials = nullptr;
        void *up_partials = nullptr;
        const bool ok =
            bindWorkspaceBuffer(&gate_partials, MoEWorkspaceBuffers::GATEUP_GATE_PARTIALS,
                                partial_count * sizeof(float), "gate/up gate partials") &&
            bindWorkspaceBuffer(&up_partials, MoEWorkspaceBuffers::GATEUP_UP_PARTIALS,
                                partial_count * sizeof(float), "gate/up up partials");
        if (!ok)
        {
            d_grouped_gateup_gate_partials_ = nullptr;
            d_grouped_gateup_up_partials_ = nullptr;
            grouped_gateup_kpart_active_cap_ = 0;
            grouped_gateup_kpart_partitions_cap_ = 0;
            grouped_gateup_kpart_intermediate_cap_ = 0;
            return false;
        }

        d_grouped_gateup_gate_partials_ = static_cast<float *>(gate_partials);
        d_grouped_gateup_up_partials_ = static_cast<float *>(up_partials);
        grouped_gateup_kpart_active_cap_ = top_k;
        grouped_gateup_kpart_partitions_cap_ = k_partitions;
        grouped_gateup_kpart_intermediate_cap_ = intermediate;
        return true;
    }

    bool CUDAMoEKernel::ensureGroupedDownKPartScratchCapacity(int k_partitions, int d_model, int slots)
    {
        // Only the discrete partition counts the kpart launcher accepts are valid.
        if (d_model <= 0 || slots <= 0 ||
            !(k_partitions == 2 || k_partitions == 4 || k_partitions == 8 ||
              k_partitions == 16))
            return false;

        // The down split-K kernel accumulates all routed experts inside each K-part
        // and writes a single [k_partitions][d_model] partial buffer.
        if (d_grouped_down_partials_ &&
            grouped_down_kpart_partitions_cap_ >= k_partitions &&
            grouped_down_kpart_d_model_cap_ >= d_model)
            return true;

        const size_t partial_count =
            static_cast<size_t>(k_partitions) *
            static_cast<size_t>(d_model);
        void *down_partials = nullptr;
        if (!bindWorkspaceBuffer(&down_partials, MoEWorkspaceBuffers::DOWN_PARTIALS,
                                 partial_count * sizeof(float), "down partials"))
        {
            d_grouped_down_partials_ = nullptr;
            grouped_down_kpart_partitions_cap_ = 0;
            grouped_down_kpart_d_model_cap_ = 0;
            grouped_down_kpart_slots_cap_ = 0;
            return false;
        }

        d_grouped_down_partials_ = static_cast<float *>(down_partials);
        grouped_down_kpart_partitions_cap_ = k_partitions;
        grouped_down_kpart_d_model_cap_ = d_model;
        grouped_down_kpart_slots_cap_ = 1;
        return true;
    }

    bool CUDAMoEKernel::ensureGroupedDownDecodeCapacity(int top_k, int intermediate)
    {
        if (top_k <= 0 || top_k > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            intermediate <= 0 || (intermediate % 32) != 0)
            return false;

        const bool need_growth = top_k > decode_down_topk_cap_ ||
                                 intermediate > decode_down_intermediate_cap_ ||
                                 !d_decode_swiglu_int8_ || !d_decode_swiglu_scales_;
        if (!need_growth)
            return true;

        const int blocks_per_row = intermediate / 32;
        void *decode_swiglu_int8 = nullptr;
        void *decode_swiglu_scales = nullptr;
        const bool ok =
            bindWorkspaceBuffer(&decode_swiglu_int8, MoEWorkspaceBuffers::DECODE_SWIGLU_INT8,
                                static_cast<size_t>(top_k) * intermediate * sizeof(int8_t), "decode SwiGLU int8") &&
            bindWorkspaceBuffer(&decode_swiglu_scales, MoEWorkspaceBuffers::DECODE_SWIGLU_SCALES,
                                static_cast<size_t>(top_k) * blocks_per_row * sizeof(float), "decode SwiGLU scales");
        if (!ok)
        {
            d_decode_swiglu_int8_ = nullptr;
            d_decode_swiglu_scales_ = nullptr;
            decode_down_topk_cap_ = 0;
            decode_down_intermediate_cap_ = 0;
            return false;
        }

        d_decode_swiglu_int8_ = static_cast<int8_t *>(decode_swiglu_int8);
        d_decode_swiglu_scales_ = static_cast<float *>(decode_swiglu_scales);
        decode_down_topk_cap_ = top_k;
        decode_down_intermediate_cap_ = intermediate;
        return true;
    }

    bool CUDAMoEKernel::ensureGroupedDecodeMetadata(
        const int *expert_ids,
        const float *expert_weights,
        int num_active,
        bool include_weights)
    {
        if (!expert_ids || num_active <= 0 ||
            num_active > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            (include_weights && !expert_weights))
        {
            return false;
        }

        const bool ids_match =
            static_cast<int>(grouped_decode_cached_expert_ids_.size()) == num_active &&
            std::equal(grouped_decode_cached_expert_ids_.begin(),
                       grouped_decode_cached_expert_ids_.end(),
                       expert_ids);
        const bool weights_match =
            !include_weights ||
            (static_cast<int>(grouped_decode_cached_weights_.size()) == num_active &&
             std::equal(grouped_decode_cached_weights_.begin(),
                        grouped_decode_cached_weights_.end(),
                        expert_weights));
        const bool capacity_ok = d_grouped_decode_expert_ids_ &&
                                 grouped_decode_metadata_cap_ >= num_active &&
                                 (!include_weights || d_grouped_decode_weights_);
        if (capacity_ok && ids_match && weights_match)
            return true;

        if (!setMoEDevice(device_ordinal_, "ensureGroupedDecodeMetadata"))
            return false;

        if (grouped_decode_metadata_cap_ < num_active || !d_grouped_decode_expert_ids_ ||
            (include_weights && !d_grouped_decode_weights_))
        {
            void *decode_expert_ids = nullptr;
            void *decode_weights = nullptr;
            const bool ok =
                bindWorkspaceBuffer(&decode_expert_ids, MoEWorkspaceBuffers::DECODE_EXPERT_IDS,
                                    static_cast<size_t>(num_active) * sizeof(int), "decode expert ids") &&
                bindWorkspaceBuffer(&decode_weights, MoEWorkspaceBuffers::DECODE_WEIGHTS,
                                    static_cast<size_t>(num_active) * sizeof(float), "decode weights");
            if (!ok)
            {
                d_grouped_decode_expert_ids_ = nullptr;
                d_grouped_decode_weights_ = nullptr;
                grouped_decode_metadata_cap_ = 0;
                return false;
            }

            d_grouped_decode_expert_ids_ = static_cast<int *>(decode_expert_ids);
            d_grouped_decode_weights_ = static_cast<float *>(decode_weights);
            grouped_decode_metadata_cap_ = num_active;
        }

        cudaStream_t stream = static_cast<cudaStream_t>(
            requireStream("CUDAMoEKernel::ensureGroupedDecodeMetadata"));
        cudaError_t err = cudaMemcpyAsync(d_grouped_decode_expert_ids_, expert_ids,
                                          static_cast<size_t>(num_active) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream);
        if (err == cudaSuccess && include_weights)
            err = cudaMemcpyAsync(d_grouped_decode_weights_, expert_weights,
                                  static_cast<size_t>(num_active) * sizeof(float),
                                  cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel] grouped decode metadata H2D failed: "
                      << cudaGetErrorString(err));
            return false;
        }

        grouped_decode_cached_expert_ids_.assign(expert_ids, expert_ids + num_active);
        if (include_weights)
            grouped_decode_cached_weights_.assign(expert_weights, expert_weights + num_active);
        else
            grouped_decode_cached_weights_.clear();
        return true;
    }

    bool CUDAMoEKernel::ensureRuntimeGateUpPointerArrays(
        int table_id,
        int top_k,
        const std::array<float *, CUDAMoEKernel::kRuntimePointerArrayMaxTopK> &gate_ptrs,
        const std::array<float *, CUDAMoEKernel::kRuntimePointerArrayMaxTopK> &up_ptrs,
        float ***d_gate_ptrs,
        float ***d_up_ptrs)
    {
        if (!d_gate_ptrs || !d_up_ptrs || table_id < 0 || top_k <= 0 ||
            top_k > static_cast<int>(kRuntimePointerArrayMaxTopK))
            return false;

        for (auto &entry : runtime_gateup_pointer_cache_)
        {
            if (entry.table_id != table_id || entry.top_k != top_k)
                continue;

            bool matches = true;
            for (int slot = 0; slot < top_k; ++slot)
            {
                if (entry.gate_ptr_values[slot] != reinterpret_cast<std::uintptr_t>(gate_ptrs[slot]) ||
                    entry.up_ptr_values[slot] != reinterpret_cast<std::uintptr_t>(up_ptrs[slot]))
                {
                    matches = false;
                    break;
                }
            }
            if (matches && entry.d_gate_ptrs && entry.d_up_ptrs)
            {
                *d_gate_ptrs = entry.d_gate_ptrs;
                *d_up_ptrs = entry.d_up_ptrs;
                return true;
            }
        }

        if (isGraphCaptureActive())
        {
            LOG_ERROR("[CUDAMoEKernel] grouped gate/up pointer cache miss during graph capture");
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "ensureRuntimeGateUpPointerArrays"))
            return false;

        RuntimeGateUpPointerCacheEntry entry;
        entry.table_id = table_id;
        entry.top_k = top_k;
        for (int slot = 0; slot < top_k; ++slot)
        {
            entry.gate_ptr_values[slot] = reinterpret_cast<std::uintptr_t>(gate_ptrs[slot]);
            entry.up_ptr_values[slot] = reinterpret_cast<std::uintptr_t>(up_ptrs[slot]);
        }

        cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&entry.d_gate_ptrs),
                                     static_cast<size_t>(top_k) * sizeof(float *));
        if (err == cudaSuccess)
            err = cudaMalloc(reinterpret_cast<void **>(&entry.d_up_ptrs),
                             static_cast<size_t>(top_k) * sizeof(float *));
        if (err == cudaSuccess)
            err = cudaMemcpyAsync(entry.d_gate_ptrs, gate_ptrs.data(),
                                  static_cast<size_t>(top_k) * sizeof(float *),
                                  cudaMemcpyHostToDevice, static_cast<cudaStream_t>(getStream()));
        if (err == cudaSuccess)
            err = cudaMemcpyAsync(entry.d_up_ptrs, up_ptrs.data(),
                                  static_cast<size_t>(top_k) * sizeof(float *),
                                  cudaMemcpyHostToDevice, static_cast<cudaStream_t>(getStream()));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel] grouped gate/up pointer cache upload failed: "
                      << cudaGetErrorString(err));
            if (entry.d_gate_ptrs)
                cudaFree(entry.d_gate_ptrs);
            if (entry.d_up_ptrs)
                cudaFree(entry.d_up_ptrs);
            return false;
        }

        runtime_gateup_pointer_cache_.push_back(entry);
        auto &cached = runtime_gateup_pointer_cache_.back();
        *d_gate_ptrs = cached.d_gate_ptrs;
        *d_up_ptrs = cached.d_up_ptrs;
        return true;
    }

    bool CUDAMoEKernel::ensureRuntimeDownPointerArrays(
        int table_id,
        int top_k,
        const std::array<const float *, CUDAMoEKernel::kRuntimePointerArrayMaxTopK> &gate_ptrs,
        const std::array<const float *, CUDAMoEKernel::kRuntimePointerArrayMaxTopK> &up_ptrs,
        const float ***d_gate_ptrs,
        const float ***d_up_ptrs)
    {
        if (!d_gate_ptrs || !d_up_ptrs || table_id < 0 || top_k <= 0 ||
            top_k > static_cast<int>(kRuntimePointerArrayMaxTopK))
            return false;

        for (auto &entry : runtime_down_pointer_cache_)
        {
            if (entry.table_id != table_id || entry.top_k != top_k)
                continue;

            bool matches = true;
            for (int slot = 0; slot < top_k; ++slot)
            {
                if (entry.gate_ptr_values[slot] != reinterpret_cast<std::uintptr_t>(gate_ptrs[slot]) ||
                    entry.up_ptr_values[slot] != reinterpret_cast<std::uintptr_t>(up_ptrs[slot]))
                {
                    matches = false;
                    break;
                }
            }
            if (matches && entry.d_gate_ptrs && entry.d_up_ptrs)
            {
                *d_gate_ptrs = entry.d_gate_ptrs;
                *d_up_ptrs = entry.d_up_ptrs;
                return true;
            }
        }

        if (isGraphCaptureActive())
        {
            LOG_ERROR("[CUDAMoEKernel] grouped down pointer cache miss during graph capture");
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "ensureRuntimeDownPointerArrays"))
            return false;

        RuntimeDownPointerCacheEntry entry;
        entry.table_id = table_id;
        entry.top_k = top_k;
        for (int slot = 0; slot < top_k; ++slot)
        {
            entry.gate_ptr_values[slot] = reinterpret_cast<std::uintptr_t>(gate_ptrs[slot]);
            entry.up_ptr_values[slot] = reinterpret_cast<std::uintptr_t>(up_ptrs[slot]);
        }

        cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&entry.d_gate_ptrs),
                                     static_cast<size_t>(top_k) * sizeof(const float *));
        if (err == cudaSuccess)
            err = cudaMalloc(reinterpret_cast<void **>(&entry.d_up_ptrs),
                             static_cast<size_t>(top_k) * sizeof(const float *));
        if (err == cudaSuccess)
            err = cudaMemcpyAsync(entry.d_gate_ptrs, gate_ptrs.data(),
                                  static_cast<size_t>(top_k) * sizeof(const float *),
                                  cudaMemcpyHostToDevice, static_cast<cudaStream_t>(getStream()));
        if (err == cudaSuccess)
            err = cudaMemcpyAsync(entry.d_up_ptrs, up_ptrs.data(),
                                  static_cast<size_t>(top_k) * sizeof(const float *),
                                  cudaMemcpyHostToDevice, static_cast<cudaStream_t>(getStream()));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel] grouped down pointer cache upload failed: "
                      << cudaGetErrorString(err));
            if (entry.d_gate_ptrs)
                cudaFree(entry.d_gate_ptrs);
            if (entry.d_up_ptrs)
                cudaFree(entry.d_up_ptrs);
            return false;
        }

        runtime_down_pointer_cache_.push_back(entry);
        auto &cached = runtime_down_pointer_cache_.back();
        *d_gate_ptrs = cached.d_gate_ptrs;
        *d_up_ptrs = cached.d_up_ptrs;
        return true;
    }

    bool CUDAMoEKernel::routeCore(const float *hidden, const void *gate_weights, TensorType gate_type,
                                  int seq_len, int d_model, int num_experts, int top_k,
                                  bool normalize_weights, DeviceRouteBuffers &buffers)
    {
        if (seq_len <= 0 || d_model <= 0 || num_experts <= 0 || top_k <= 0 || top_k > num_experts)
            return false;
        const bool gate_is_fp32 = (gate_type == TensorType::FP32);
        const bool gate_is_bf16 = (gate_type == TensorType::BF16);
        if (!gate_is_fp32 && !gate_is_bf16)
        {
            LOG_ERROR("[CUDAMoEKernel] CUDA router supports FP32 or BF16 gate weights, got tensor type "
                      << static_cast<int>(gate_type));
            return false;
        }
        if (num_experts > 1024 || top_k > static_cast<int>(kDeviceMoEMaxTopK))
        {
            LOG_ERROR("[CUDAMoEKernel] Unsupported routing shape num_experts=" << num_experts
                                                                                << " top_k=" << top_k);
            return false;
        }

        const size_t logits_count = static_cast<size_t>(seq_len) * num_experts;
        const size_t topk_count = static_cast<size_t>(seq_len) * top_k;
        if (!ensureRouteBufferCapacity(logits_count, topk_count))
            return false;

        void *stream = getStream();
        if (!requireAlignedPointer(hidden, 16, "hidden", "routeCore") ||
            !requireAlignedPointer(d_route_logits_, 16, "route logits", "routeCore"))
            return false;
        if (gate_is_fp32 && !requireAlignedPointer(gate_weights, 16, "FP32 gate", "routeCore"))
            return false;
        if (!requireCudaDevicePointer(hidden, device_ordinal_, "hidden", "routeCore", stream) ||
            !requireCudaDevicePointer(gate_weights, device_ordinal_, "gate weights", "routeCore", stream) ||
            !requireCudaDevicePointer(d_route_logits_, device_ordinal_, "route logits", "routeCore", stream))
            return false;

        const bool use_cublas_small_m =
            gate_is_fp32 && seq_len >= 2 && seq_len <= 4;
        const bool route_ok = use_cublas_small_m
                                  ? routeLogitsCuBLAS(hidden, static_cast<const float *>(gate_weights), d_route_logits_,
                                                      seq_len, d_model, num_experts)
                                  : (gate_is_fp32
                                         ? cudaMoE_route_logits(hidden, static_cast<const float *>(gate_weights), d_route_logits_,
                                                                seq_len, d_model, num_experts,
                                                                device_ordinal_, stream)
                                         : cudaMoE_route_logits_bf16(hidden, gate_weights, d_route_logits_,
                                                                     seq_len, d_model, num_experts,
                                                                     device_ordinal_, stream));
        if (!route_ok)
            return false;
        if (use_cublas_small_m)
        {
            PerfStatsCollector::addCounter(
                "kernel", "cuda_moe_router_cublas_small_m_calls", 1.0, {}, {},
                {{"seq_len", std::to_string(seq_len)},
                 {"d_model", std::to_string(d_model)},
                 {"num_experts", std::to_string(num_experts)}});
        }
        if (gate_is_fp32 && seq_len >= 16)
        {
            PerfStatsCollector::addCounter(
                "kernel", "cuda_moe_router_cublas_prefill_calls", 1.0, {}, {},
                {{"seq_len", std::to_string(seq_len)},
                 {"d_model", std::to_string(d_model)},
                 {"num_experts", std::to_string(num_experts)}});
        }
        if (!cudaMoE_softmax_topk(d_route_logits_, d_route_indices_, d_route_weights_,
                                  seq_len, num_experts, top_k, normalize_weights,
                                  device_ordinal_, getStream()))
            return false;

        buffers.d_logits = d_route_logits_;
        buffers.d_indices = d_route_indices_;
        buffers.d_weights = d_route_weights_;
        buffers.logits_count = logits_count;
        buffers.topk_count = topk_count;
        return true;
    }

    bool CUDAMoEKernel::route(const float *hidden, const float *gate_weights,
                              int seq_len, int d_model, int num_experts, int top_k,
                              bool normalize_weights, MoERoutingResult &result)
    {
        result.expert_indices.resize(static_cast<size_t>(seq_len) * top_k);
        result.expert_weights.resize(static_cast<size_t>(seq_len) * top_k);
        result.router_logits.resize(static_cast<size_t>(seq_len) * num_experts);

        std::vector<int> indices(num_experts);
        for (int token = 0; token < seq_len; ++token)
        {
            const float *row = hidden + static_cast<size_t>(token) * d_model;
            float *logits = result.router_logits.data() + static_cast<size_t>(token) * num_experts;
            for (int expert = 0; expert < num_experts; ++expert)
            {
                const float *gate = gate_weights + static_cast<size_t>(expert) * d_model;
                float dot = 0.0f;
                for (int i = 0; i < d_model; ++i)
                    dot += row[i] * gate[i];
                logits[expert] = dot;
            }

            const float max_logit = *std::max_element(logits, logits + num_experts);
            float sum = 0.0f;
            for (int expert = 0; expert < num_experts; ++expert)
            {
                logits[expert] = std::exp(logits[expert] - max_logit);
                sum += logits[expert];
            }
            for (int expert = 0; expert < num_experts; ++expert)
                logits[expert] = (sum > 0.0f) ? logits[expert] / sum : 0.0f;

            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                              [logits](int lhs, int rhs)
                              { return logits[lhs] > logits[rhs]; });

            float topk_sum = 0.0f;
            for (int k = 0; k < top_k; ++k)
                topk_sum += logits[indices[k]];

            for (int k = 0; k < top_k; ++k)
            {
                const size_t out = static_cast<size_t>(token) * top_k + k;
                result.expert_indices[out] = indices[k];
                result.expert_weights[out] = normalize_weights && topk_sum > 0.0f
                                                 ? logits[indices[k]] / topk_sum
                                                 : logits[indices[k]];
            }
        }
        return true;
    }

    void CUDAMoEKernel::gatherTokenBatch(const float *hidden, float *batch_buffer,
                                         const int *token_indices, int num_tokens, int d_model)
    {
        if (num_tokens <= 0)
            return;
        cudaMoE_gather_tokens(hidden, batch_buffer, token_indices,
                              num_tokens, d_model, device_ordinal_, getStream());
    }

    void CUDAMoEKernel::scatterAddWeighted(float *output, const float *expert_output,
                                           const int *token_indices, const float *weights,
                                           int num_tokens, int d_model)
    {
        if (num_tokens <= 0)
            return;
        cudaMoE_scatter_add(output, expert_output, token_indices, weights,
                            num_tokens, d_model, device_ordinal_, getStream());
    }

    void CUDAMoEKernel::sharedExpertGate(const float *input, const float *gate_inp,
                                         float *shared_output, int seq_len, int d_model)
    {
        if (seq_len <= 0)
            return;
        cudaMoE_shared_expert_gate(input, gate_inp, shared_output,
                                   seq_len, d_model, device_ordinal_, getStream());
    }

    void CUDAMoEKernel::swiGLU(float *gate, const float *up, int count)
    {
        if (count <= 0)
            return;
        cudaMoE_swiglu(gate, up, count, device_ordinal_, getStream());
    }

    void CUDAMoEKernel::weightedAdd(float *output, const float *input, float weight, int count)
    {
        if (count <= 0)
            return;
        cudaMoE_weighted_add(output, input, weight, count, device_ordinal_, getStream());
    }

    bool CUDAMoEKernel::routeWithTensors(ITensor *hidden, ITensor *gate_weights,
                                         int seq_len, int d_model, int num_experts, int top_k,
                                         bool normalize_weights,
                                         ITensor *output_indices, ITensor *output_weights,
                                         MoERoutingResult &host_result)
    {
        void *stream = requireStream("CUDAMoEKernel::routeWithTensors");
        const DeviceId device = deviceId();
        if (!ensureTensorOnDevice(hidden, device, stream, "hidden") ||
            !ensureTensorOnDevice(gate_weights, device, stream, "gate_weights") ||
            !ensureOutputOnDevice(output_indices, device, stream, "output_indices") ||
            !ensureOutputOnDevice(output_weights, device, stream, "output_weights"))
            return false;

        if (seq_len <= 0 || d_model <= 0 || num_experts <= 0 || top_k <= 0 || top_k > num_experts)
        {
            LOG_ERROR("[CUDAMoEKernel::routeWithTensors] invalid routing shape seq_len="
                      << seq_len << " d_model=" << d_model
                      << " num_experts=" << num_experts << " top_k=" << top_k);
            return false;
        }

        auto *hidden_base = asTensorBase(hidden, "routeWithTensors hidden");
        auto *gate_base = asTensorBase(gate_weights, "routeWithTensors gate_weights");
        auto *indices_base = asTensorBase(output_indices, "routeWithTensors output_indices");
        auto *weights_base = asTensorBase(output_weights, "routeWithTensors output_weights");
        if (!hidden_base || !gate_base || !indices_base || !weights_base)
            return false;

        if (!requireTensorType(hidden_base, TensorType::FP32, "hidden", "routeWithTensors") ||
            !requireTensorTypeOneOf(gate_base, TensorType::FP32, TensorType::BF16, "gate_weights", "routeWithTensors") ||
            !requireTensorType(indices_base, TensorType::FP32, "output_indices", "routeWithTensors") ||
            !requireTensorType(weights_base, TensorType::FP32, "output_weights", "routeWithTensors"))
            return false;

        const size_t required_hidden = static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
        const size_t required_gate = static_cast<size_t>(num_experts) * static_cast<size_t>(d_model);
        const size_t required_topk = static_cast<size_t>(seq_len) * static_cast<size_t>(top_k);
        if (!requireMatrixCapacity(hidden, seq_len, d_model, "hidden", "routeWithTensors") ||
            !requireMatrixCapacity(gate_weights, num_experts, d_model, "gate_weights", "routeWithTensors") ||
            !requireTensorElements(hidden, required_hidden, "hidden", "routeWithTensors") ||
            !requireTensorElements(gate_weights, required_gate, "gate_weights", "routeWithTensors") ||
            !requireTensorElements(output_indices, required_topk, "output_indices", "routeWithTensors") ||
            !requireTensorElements(output_weights, required_topk, "output_weights", "routeWithTensors"))
            return false;

        const float *d_hidden = static_cast<const float *>(hidden->gpu_data_ptr());
        const void *d_gate = gate_weights->gpu_data_ptr();
        float *d_idx = static_cast<float *>(output_indices->gpu_data_ptr());
        float *d_wt = static_cast<float *>(output_weights->gpu_data_ptr());
        if (!d_hidden || !d_gate || !d_idx || !d_wt)
            return false;

        DeviceRouteBuffers buffers;
        if (!routeCore(d_hidden, d_gate, gate_base->native_type(),
                       seq_len, d_model, num_experts, top_k, normalize_weights, buffers))
            return false;

        if (!cudaMoE_int_to_float(buffers.d_indices, d_idx, static_cast<int>(buffers.topk_count), device_ordinal_, stream))
            return false;
        cudaError_t err = cudaMemcpyAsync(d_wt, buffers.d_weights,
                                          buffers.topk_count * sizeof(float),
                                          cudaMemcpyDeviceToDevice,
                                          static_cast<cudaStream_t>(stream));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel::routeWithTensors] D2D weights failed: " << cudaGetErrorString(err));
            return false;
        }

        host_result.expert_indices.clear();
        host_result.expert_weights.clear();
        host_result.router_logits.clear();

        float *h_idx = nullptr;
        float *h_wt = nullptr;
        const bool capture_active = isGraphCaptureActive() || isCudaStreamCapturing(stream);
        const bool needs_decode_host_topk = (seq_len == 1);
#ifdef ENABLE_PIPELINE_SNAPSHOTS
        const bool needs_snapshot_host_topk = true;
#else
        const bool needs_snapshot_host_topk = false;
#endif
        const bool needs_host_topk = !capture_active && (needs_decode_host_topk || needs_snapshot_host_topk);
        if (needs_host_topk)
        {
            h_idx = static_cast<float *>(output_indices->raw_mutable_data());
            h_wt = static_cast<float *>(output_weights->raw_mutable_data());
            if (!h_idx || !h_wt)
                return false;

            err = cudaMemcpyAsync(h_idx, d_idx, buffers.topk_count * sizeof(float),
                                  cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream));
            if (err != cudaSuccess)
                return false;
            err = cudaMemcpyAsync(h_wt, d_wt, buffers.topk_count * sizeof(float),
                                  cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream));
            if (err != cudaSuccess)
                return false;
        }

#ifdef ENABLE_PIPELINE_SNAPSHOTS
        if (!capture_active)
        {
            host_result.router_logits.resize(buffers.logits_count);
            err = cudaMemcpyAsync(host_result.router_logits.data(), buffers.d_logits,
                                  buffers.logits_count * sizeof(float),
                                  cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream));
            if (err != cudaSuccess)
                return false;
        }
#endif

        if (needs_host_topk || !host_result.router_logits.empty())
        {
            err = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
            if (err != cudaSuccess)
            {
                LOG_ERROR("[CUDAMoEKernel::routeWithTensors] stream sync failed: " << cudaGetErrorString(err));
                return false;
            }
        }

        if (needs_host_topk)
        {
            host_result.expert_indices.resize(buffers.topk_count);
            host_result.expert_weights.assign(h_wt, h_wt + buffers.topk_count);
            for (size_t i = 0; i < buffers.topk_count; ++i)
                host_result.expert_indices[i] = static_cast<int>(h_idx[i]);
            markSynced(output_indices);
            markSynced(output_weights);
        }
        else
        {
            markDeviceWritten(output_indices, device, stream);
            markDeviceWritten(output_weights, device, stream);
        }

        return true;
    }

    bool CUDAMoEKernel::decodeRouteSelect(DeviceMoELayerRuntime *runtime_layer,
                                          ITensor *hidden, ITensor *gate_weights,
                                          int d_model, int num_experts, int top_k,
                                          bool normalize_weights,
                                          ITensor *output_indices, ITensor *output_weights,
                                          bool write_legacy_outputs,
                                          bool update_runtime_histogram)
    {
        void *stream = requireStream("CUDAMoEKernel::decodeRouteSelect");
        const DeviceId device = deviceId();
        if (!runtime_layer || !hidden || !gate_weights)
            return false;
        if (!ensureTensorOnDevice(hidden, device, stream, "hidden") ||
            !ensureTensorOnDevice(gate_weights, device, stream, "gate_weights"))
            return false;

        auto *hidden_base = asTensorBase(hidden, "decodeRouteSelect hidden");
        auto *gate_base = asTensorBase(gate_weights, "decodeRouteSelect gate_weights");
        if (!hidden_base || !gate_base)
            return false;
        if (hidden_base->native_type() != TensorType::FP32)
        {
            LOG_ERROR("[CUDAMoEKernel::decodeRouteSelect] hidden must be FP32, got "
                      << tensorTypeName(hidden_base->native_type()));
            return false;
        }
        const TensorType gate_type = gate_base->native_type();
        const bool gate_is_fp32 = (gate_type == TensorType::FP32);
        const bool gate_is_bf16 = (gate_type == TensorType::BF16);
        if (!gate_is_fp32 && !gate_is_bf16)
        {
            LOG_ERROR("[CUDAMoEKernel::decodeRouteSelect] CUDA runtime router supports FP32 or BF16 gate weights, got "
                      << tensorTypeName(gate_type));
            return false;
        }

        float *legacy_indices = nullptr;
        float *legacy_weights = nullptr;
        if (write_legacy_outputs)
        {
            if (!output_indices || !output_weights ||
                !ensureOutputOnDevice(output_indices, device, stream, "output_indices") ||
                !ensureOutputOnDevice(output_weights, device, stream, "output_weights"))
                return false;
            legacy_indices = static_cast<float *>(output_indices->gpu_data_ptr());
            legacy_weights = static_cast<float *>(output_weights->gpu_data_ptr());
        }

        const float *d_hidden = static_cast<const float *>(hidden->gpu_data_ptr());
        const void *d_gate = gate_weights->gpu_data_ptr();
        if (!d_hidden || !d_gate)
            return false;

        if (!ensureRouteBufferCapacity(static_cast<size_t>(num_experts), static_cast<size_t>(top_k)))
            return false;
        if (!requireAlignedPointer(d_hidden, 16, "hidden", "decodeRouteSelect") ||
            !requireAlignedPointer(d_route_logits_, 16, "route logits", "decodeRouteSelect"))
            return false;
        if (gate_is_fp32 && !requireAlignedPointer(d_gate, 16, "FP32 gate", "decodeRouteSelect"))
            return false;

        const bool route_ok = gate_is_fp32
                                  ? cudaMoE_route_logits(d_hidden, static_cast<const float *>(d_gate), d_route_logits_,
                                                         /*seq_len=*/1, d_model, num_experts,
                                                         device_ordinal_, stream)
                                  : cudaMoE_route_logits_bf16(d_hidden, d_gate, d_route_logits_,
                                                              /*seq_len=*/1, d_model, num_experts,
                                                              device_ordinal_, stream);
        if (!route_ok)
            return false;
        if (!cudaMoE_softmax_topk_decode_runtime(d_route_logits_,
                                                 runtimeTopKExpertIdsDevice(runtime_layer),
                                                 runtimeTopKWeightsDevice(runtime_layer),
                                                 runtimeDecodeHistogramDevice(runtime_layer),
                                                 legacy_indices, legacy_weights,
                                                 num_experts, top_k, normalize_weights,
                                                 write_legacy_outputs, update_runtime_histogram,
                                                 device_ordinal_, stream))
            return false;

        if (write_legacy_outputs)
        {
            markDeviceWritten(output_indices, device, stream);
            markDeviceWritten(output_weights, device, stream);
        }
        return true;
    }

    void CUDAMoEKernel::zeroBuffer(ITensor *tensor, size_t bytes)
    {
        void *stream = requireStream("CUDAMoEKernel::zeroBuffer");
        if (!ensureOutputOnDevice(tensor, deviceId(), stream, "zeroBuffer"))
            return;
        void *ptr = tensor->gpu_data_ptr();
        cudaError_t err = cudaMemsetAsync(ptr, 0, bytes, static_cast<cudaStream_t>(stream));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel::zeroBuffer] cudaMemsetAsync failed: " << cudaGetErrorString(err));
            return;
        }
        markDeviceWritten(tensor, deviceId(), stream);
    }

    void CUDAMoEKernel::gatherTokenBatchFromTensors(ITensor *hidden, ITensor *batch_buffer,
                                                    const int *host_token_indices, int num_tokens, int d_model)
    {
        if (num_tokens <= 0)
            return;
        void *stream = requireStream("CUDAMoEKernel::gatherTokenBatchFromTensors");
        const DeviceId device = deviceId();
        if (!ensureTensorOnDevice(hidden, device, stream, "hidden") ||
            !ensureOutputOnDevice(batch_buffer, device, stream, "batch_buffer") ||
            !ensureStagingCapacity(num_tokens))
            return;

        cudaError_t err = cudaMemcpyAsync(d_staging_indices_, host_token_indices,
                                          static_cast<size_t>(num_tokens) * sizeof(int),
                                          cudaMemcpyHostToDevice,
                                          static_cast<cudaStream_t>(stream));
        if (err != cudaSuccess)
            return;
        gatherTokenBatch(static_cast<const float *>(hidden->gpu_data_ptr()),
                         static_cast<float *>(batch_buffer->gpu_data_ptr()),
                         d_staging_indices_, num_tokens, d_model);
        markDeviceWritten(batch_buffer, device, stream);
    }

    void CUDAMoEKernel::scatterAddWeightedFromTensors(ITensor *output, ITensor *expert_output,
                                                      const int *host_token_indices, const float *host_weights,
                                                      int num_tokens, int d_model)
    {
        if (num_tokens <= 0)
            return;
        void *stream = requireStream("CUDAMoEKernel::scatterAddWeightedFromTensors");
        const DeviceId device = deviceId();
        if (!ensureTensorOnDevice(output, device, stream, "output") ||
            !ensureTensorOnDevice(expert_output, device, stream, "expert_output") ||
            !ensureStagingCapacity(num_tokens))
            return;

        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        cudaError_t err = cudaMemcpyAsync(d_staging_indices_, host_token_indices,
                                          static_cast<size_t>(num_tokens) * sizeof(int),
                                          cudaMemcpyHostToDevice, cuda_stream);
        if (err != cudaSuccess)
            return;
        err = cudaMemcpyAsync(d_staging_weights_, host_weights,
                              static_cast<size_t>(num_tokens) * sizeof(float),
                              cudaMemcpyHostToDevice, cuda_stream);
        if (err != cudaSuccess)
            return;

        scatterAddWeighted(static_cast<float *>(output->gpu_data_ptr()),
                           static_cast<const float *>(expert_output->gpu_data_ptr()),
                           d_staging_indices_, d_staging_weights_, num_tokens, d_model);
        markDeviceWritten(output, device, stream);
    }

    void CUDAMoEKernel::sharedExpertGateFromTensors(ITensor *input, ITensor *gate_inp, ITensor *shared_output,
                                                    int seq_len, int d_model)
    {
        if (seq_len <= 0)
            return;
        void *stream = requireStream("CUDAMoEKernel::sharedExpertGateFromTensors");
        const DeviceId device = deviceId();
        if (!ensureTensorOnDevice(input, device, stream, "input") ||
            !ensureTensorOnDevice(gate_inp, device, stream, "gate_inp") ||
            !ensureTensorOnDevice(shared_output, device, stream, "shared_output"))
            return;
        sharedExpertGate(static_cast<const float *>(input->gpu_data_ptr()),
                         static_cast<const float *>(gate_inp->gpu_data_ptr()),
                         static_cast<float *>(shared_output->gpu_data_ptr()),
                         seq_len, d_model);
        markDeviceWritten(shared_output, device, stream);
    }

    void CUDAMoEKernel::swiGLUFromTensors(ITensor *gate, ITensor *up, int count)
    {
        if (count <= 0)
            return;
        void *stream = requireStream("CUDAMoEKernel::swiGLUFromTensors");
        const DeviceId device = deviceId();
        if (!ensureTensorOnDevice(gate, device, stream, "gate") ||
            !ensureTensorOnDevice(up, device, stream, "up"))
            return;
        swiGLU(static_cast<float *>(gate->gpu_data_ptr()),
               static_cast<const float *>(up->gpu_data_ptr()), count);
        markDeviceWritten(gate, device, stream);
    }

    void CUDAMoEKernel::weightedAddFromTensors(ITensor *output, ITensor *input, float weight, int count)
    {
        if (count <= 0)
            return;
        void *stream = requireStream("CUDAMoEKernel::weightedAddFromTensors");
        const DeviceId device = deviceId();
        if (!ensureTensorOnDevice(output, device, stream, "output") ||
            !ensureTensorOnDevice(input, device, stream, "input"))
            return;
        weightedAdd(static_cast<float *>(output->gpu_data_ptr()),
                    static_cast<const float *>(input->gpu_data_ptr()), weight, count);
        markDeviceWritten(output, device, stream);
    }

    bool CUDAMoEKernel::groupTokensByExpertDevice(const int *d_routing_indices,
                                                  const float *d_routing_weights,
                                                  int seq_len, int num_experts, int top_k,
                                                  int *d_expert_offsets, int *d_expert_counts,
                                                  int *d_grouped_token_indices, float *d_grouped_weights)
    {
        const int total_slots = seq_len * top_k;
        void *stream = requireStream("CUDAMoEKernel::groupTokensByExpertDevice");
        cudaError_t err = cudaMemsetAsync(d_expert_counts, 0, static_cast<size_t>(num_experts) * sizeof(int),
                                          static_cast<cudaStream_t>(stream));
        if (err != cudaSuccess)
            return false;
        err = cudaMemsetAsync(d_group_write_heads_, 0, static_cast<size_t>(num_experts) * sizeof(int),
                              static_cast<cudaStream_t>(stream));
        if (err != cudaSuccess)
            return false;
        return cudaMoE_count_per_expert(d_routing_indices, d_expert_counts, total_slots,
                                        num_experts, device_ordinal_, stream) &&
               cudaMoE_exclusive_scan(d_expert_counts, d_expert_offsets,
                                       num_experts, device_ordinal_, stream) &&
               cudaMoE_scatter_tokens_deterministic(
                   d_routing_indices, d_routing_weights,
                   d_expert_offsets, d_expert_counts,
                   d_grouped_token_indices,
                   d_group_original_to_grouped_,
                   d_group_original_expert_ids_,
                   d_grouped_weights,
                   total_slots, top_k, num_experts,
                   device_ordinal_, stream);
    }

    bool CUDAMoEKernel::prepareExpertGroups(ITensor *routing_indices, ITensor *routing_weights,
                                            int seq_len, int num_experts, int top_k)
    {
        if (seq_len <= 0 || num_experts <= 0 || top_k <= 0)
            return false;
        void *stream = requireStream("CUDAMoEKernel::prepareExpertGroups");
        const DeviceId device = deviceId();
        const int total_slots = seq_len * top_k;
        if (!ensureTensorOnDevice(routing_indices, device, stream, "routing_indices") ||
            !ensureTensorOnDevice(routing_weights, device, stream, "routing_weights") ||
            !ensureGroupingBufferCapacity(total_slots, num_experts))
            return false;

        const float *d_float_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        const float *d_float_weights = static_cast<const float *>(routing_weights->gpu_data_ptr());
        if (!d_float_indices || !d_float_weights)
            return false;

        if (!cudaMoE_float_to_int(d_float_indices, d_group_int_indices_, total_slots, device_ordinal_, stream))
            return false;
        if (!groupTokensByExpertDevice(d_group_int_indices_, d_float_weights,
                                       seq_len, num_experts, top_k,
                                       d_group_offsets_, d_group_counts_,
                                       d_group_token_indices_, d_group_weights_))
            return false;

        host_expert_counts_.resize(num_experts);
        host_expert_offsets_.resize(num_experts);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        cudaError_t err = cudaMemcpyAsync(host_expert_counts_.data(), d_group_counts_,
                                          static_cast<size_t>(num_experts) * sizeof(int),
                                          cudaMemcpyDeviceToHost, cuda_stream);
        if (err != cudaSuccess)
            return false;
        err = cudaMemcpyAsync(host_expert_offsets_.data(), d_group_offsets_,
                              static_cast<size_t>(num_experts) * sizeof(int),
                              cudaMemcpyDeviceToHost, cuda_stream);
        if (err != cudaSuccess)
            return false;
        err = cudaStreamSynchronize(cuda_stream);
        if (err != cudaSuccess)
            return false;

        prepared_num_experts_ = num_experts;
        return true;
    }

    int CUDAMoEKernel::uploadGroupedExpertDownDescriptorTable(
        const DeviceNativeVNNIMatrixDesc *down_descs,
        int num_experts,
        int d_model,
        int intermediate)
    {
        if (!down_descs || num_experts <= 0 || d_model <= 0 || intermediate <= 0 || (intermediate % 32) != 0)
            return -1;
        if (!setMoEDevice(device_ordinal_, "uploadGroupedExpertDownDescriptorTable"))
            return -1;

        uint8_t codebook_id = 0;
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &desc = down_descs[expert_id];
            if (!desc.valid())
            {
                LOG_DEBUG("[CUDAMoEKernel::uploadGroupedExpertDownDescriptorTable] Invalid descriptor for expert "
                          << expert_id);
                return -1;
            }
            if (expert_id == 0)
            {
                codebook_id = desc.codebook_id;
                if (!cudaGroupedPrefillSupportsCodebook(codebook_id))
                {
                    LOG_DEBUG("[CUDAMoEKernel::uploadGroupedExpertDownDescriptorTable] Unsupported codebook "
                              << static_cast<int>(codebook_id));
                    return -1;
                }
            }
            if (!validateCudaGroupedDesc(desc, d_model, intermediate, codebook_id))
            {
                LOG_DEBUG("[CUDAMoEKernel::uploadGroupedExpertDownDescriptorTable] Descriptor shape mismatch for expert "
                          << expert_id);
                return -1;
            }
        }

        if (codebook_id >= 11 && codebook_id <= 17)
        {
            static std::mutex iq_table_mutex;
            static std::unordered_set<int> iq_init_devices;
            std::lock_guard<std::mutex> lock(iq_table_mutex);
            if (!iq_init_devices.count(device_ordinal_))
            {
                if (!cudaNativeVNNIInitIQGridTables_tuned())
                {
                    LOG_ERROR("[CUDAMoEKernel::uploadGroupedExpertDownDescriptorTable] IQ grid table init failed");
                    return -1;
                }
                iq_init_devices.insert(device_ordinal_);
            }
        }

        GroupedDownDescriptorTable table;
        table.host_descs.assign(down_descs, down_descs + num_experts);
        table.num_experts = num_experts;
        table.d_model = d_model;
        table.intermediate = intermediate;
        table.codebook_id = codebook_id;

        DeviceNativeVNNIMatrixDesc *device_descs = nullptr;
        cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&device_descs),
                                     static_cast<size_t>(num_experts) * sizeof(DeviceNativeVNNIMatrixDesc));
        if (err == cudaSuccess)
            err = cudaMemcpyAsync(device_descs, table.host_descs.data(),
                                  static_cast<size_t>(num_experts) * sizeof(DeviceNativeVNNIMatrixDesc),
                                  cudaMemcpyHostToDevice, static_cast<cudaStream_t>(getStream()));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel::uploadGroupedExpertDownDescriptorTable] descriptor upload failed: "
                      << cudaGetErrorString(err));
            if (device_descs)
                cudaFree(device_descs);
            return -1;
        }

        table.device_descs = device_descs;
        table.valid = true;
        grouped_down_desc_tables_.push_back(std::move(table));
        return static_cast<int>(grouped_down_desc_tables_.size() - 1);
    }

    int CUDAMoEKernel::uploadGroupedExpertGateUpDescriptorTables(
        const DeviceNativeVNNIMatrixDesc *gate_descs,
        const DeviceNativeVNNIMatrixDesc *up_descs,
        int num_experts,
        int d_model,
        int intermediate)
    {
        if (!gate_descs || !up_descs || num_experts <= 0 || d_model <= 0 || intermediate <= 0 || (d_model % 32) != 0)
            return -1;
        if (!setMoEDevice(device_ordinal_, "uploadGroupedExpertGateUpDescriptorTables"))
            return -1;

        uint8_t codebook_id = 0;
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &gate_desc = gate_descs[expert_id];
            const auto &up_desc = up_descs[expert_id];
            if (!gate_desc.valid() || !up_desc.valid() || gate_desc.codebook_id != up_desc.codebook_id)
            {
                LOG_DEBUG("[CUDAMoEKernel::uploadGroupedExpertGateUpDescriptorTables] Invalid gate/up descriptor pair for expert "
                          << expert_id);
                return -1;
            }
            if (expert_id == 0)
            {
                codebook_id = gate_desc.codebook_id;
                if (!cudaGroupedPrefillSupportsCodebook(codebook_id))
                {
                    LOG_DEBUG("[CUDAMoEKernel::uploadGroupedExpertGateUpDescriptorTables] Unsupported codebook "
                              << static_cast<int>(codebook_id));
                    return -1;
                }
            }
            if (!validateCudaGroupedDesc(gate_desc, intermediate, d_model, codebook_id) ||
                !validateCudaGroupedDesc(up_desc, intermediate, d_model, codebook_id))
            {
                LOG_DEBUG("[CUDAMoEKernel::uploadGroupedExpertGateUpDescriptorTables] Descriptor shape mismatch for expert "
                          << expert_id);
                return -1;
            }
        }

        if (codebook_id >= 11 && codebook_id <= 17)
        {
            static std::mutex iq_table_mutex;
            static std::unordered_set<int> iq_init_devices;
            std::lock_guard<std::mutex> lock(iq_table_mutex);
            if (!iq_init_devices.count(device_ordinal_))
            {
                if (!cudaNativeVNNIInitIQGridTables_tuned())
                {
                    LOG_ERROR("[CUDAMoEKernel::uploadGroupedExpertGateUpDescriptorTables] IQ grid table init failed");
                    return -1;
                }
                iq_init_devices.insert(device_ordinal_);
            }
        }

        GroupedGateUpDescriptorTable table;
        table.host_gate_descs.assign(gate_descs, gate_descs + num_experts);
        table.host_up_descs.assign(up_descs, up_descs + num_experts);
        table.num_experts = num_experts;
        table.d_model = d_model;
        table.intermediate = intermediate;
        table.codebook_id = codebook_id;

        DeviceNativeVNNIMatrixDesc *device_gate_descs = nullptr;
        DeviceNativeVNNIMatrixDesc *device_up_descs = nullptr;
        cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&device_gate_descs),
                                     static_cast<size_t>(num_experts) * sizeof(DeviceNativeVNNIMatrixDesc));
        if (err == cudaSuccess)
            err = cudaMalloc(reinterpret_cast<void **>(&device_up_descs),
                             static_cast<size_t>(num_experts) * sizeof(DeviceNativeVNNIMatrixDesc));
        if (err == cudaSuccess)
            err = cudaMemcpyAsync(device_gate_descs, table.host_gate_descs.data(),
                                  static_cast<size_t>(num_experts) * sizeof(DeviceNativeVNNIMatrixDesc),
                                  cudaMemcpyHostToDevice, static_cast<cudaStream_t>(getStream()));
        if (err == cudaSuccess)
            err = cudaMemcpyAsync(device_up_descs, table.host_up_descs.data(),
                                  static_cast<size_t>(num_experts) * sizeof(DeviceNativeVNNIMatrixDesc),
                                  cudaMemcpyHostToDevice, static_cast<cudaStream_t>(getStream()));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel::uploadGroupedExpertGateUpDescriptorTables] descriptor upload failed: "
                      << cudaGetErrorString(err));
            if (device_gate_descs)
                cudaFree(device_gate_descs);
            if (device_up_descs)
                cudaFree(device_up_descs);
            return -1;
        }

        table.device_gate_descs = device_gate_descs;
        table.device_up_descs = device_up_descs;
        table.valid = true;
        grouped_gateup_desc_tables_.push_back(std::move(table));
        return static_cast<int>(grouped_gateup_desc_tables_.size() - 1);
    }

    bool CUDAMoEKernel::prepareExpertGroupsAsync(ITensor *routing_indices, ITensor *routing_weights,
                                                 int seq_len, int num_experts, int top_k)
    {
        if (seq_len <= 0 || num_experts <= 0 || top_k <= 0)
            return false;
        void *stream = requireStream("CUDAMoEKernel::prepareExpertGroupsAsync");
        const DeviceId device = deviceId();
        const int total_slots = seq_len * top_k;
        if (!ensureTensorOnDevice(routing_indices, device, stream, "routing_indices") ||
            !ensureTensorOnDevice(routing_weights, device, stream, "routing_weights") ||
            !ensureGroupingBufferCapacity(total_slots, num_experts))
            return false;

        const float *d_float_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        const float *d_float_weights = static_cast<const float *>(routing_weights->gpu_data_ptr());
        if (!d_float_indices || !d_float_weights)
            return false;

        group_active_expert_slots_ = 0;
        const bool verifier_sized_group =
            total_slots <= 64 &&
            seq_len <= 4 &&
            num_experts <= static_cast<int>(kDeviceMoEMaxExperts) &&
            top_k <= static_cast<int>(kDeviceMoEMaxTopK);
        if (verifier_sized_group)
        {
            const int active_expert_slots = std::min(total_slots, num_experts);
            if (!cudaMoE_group_tokens_small_float(
                    d_float_indices,
                    d_float_weights,
                    d_group_counts_,
                    d_group_offsets_,
                    d_group_token_indices_,
                    d_group_original_to_grouped_,
                    d_group_original_expert_ids_,
                    d_group_weights_,
                    d_group_active_expert_ids_,
                    total_slots,
                    num_experts,
                    top_k,
                    active_expert_slots,
                    device_ordinal_,
                    stream))
            {
                return false;
            }

            group_active_expert_slots_ = active_expert_slots;
            prepared_num_experts_ = num_experts;
            PerfStatsCollector::addCounter(
                "kernel", "cuda_moe_small_prefill_grouping_calls", 1.0, {}, {},
                {{"seq_len", std::to_string(seq_len)},
                 {"top_k", std::to_string(top_k)},
                 {"num_experts", std::to_string(num_experts)},
                 {"total_slots", std::to_string(total_slots)}});
            return true;
        }

        if (!cudaMoE_float_to_int(d_float_indices, d_group_int_indices_, total_slots, device_ordinal_, stream))
            return false;
        if (!groupTokensByExpertDevice(d_group_int_indices_, d_float_weights,
                                       seq_len, num_experts, top_k,
                                       d_group_offsets_, d_group_counts_,
                                       d_group_token_indices_, d_group_weights_))
            return false;

        prepared_num_experts_ = num_experts;
        return true;
    }

    bool CUDAMoEKernel::prepareSharedExpertPrefillGroup(int seq_len)
    {
        if (seq_len <= 0)
            return false;
        void *stream = requireStream("CUDAMoEKernel::prepareSharedExpertPrefillGroup");
        if (!ensureGroupingBufferCapacity(seq_len, /*num_experts=*/1))
            return false;
        if (!cudaMoE_prepare_shared_expert_group(
                d_group_offsets_,
                d_group_counts_,
                d_group_token_indices_,
                d_group_weights_,
                d_group_active_expert_ids_,
                seq_len,
                device_ordinal_,
                stream))
        {
            return false;
        }
        prepared_num_experts_ = 1;
        group_active_expert_slots_ = 1;
        PerfStatsCollector::addCounter(
            "kernel", "cuda_moe_shared_expert_prefill_group_calls", 1.0, {}, {},
            {{"seq_len", std::to_string(seq_len)},
             {"top_k", "1"},
             {"active_expert_slots", "1"}});
        return true;
    }

    bool CUDAMoEKernel::executeGroupedPrefillPipeline(
        ITensor *hidden, ITensor *output,
        int gateup_desc_table_id,
        int down_desc_table_id,
        int seq_len, int d_model, int intermediate,
        int num_experts, int top_k)
    {
        if (seq_len <= 0 || d_model <= 0 || intermediate <= 0 || num_experts <= 0 || top_k <= 0)
            return false;
        if (gateup_desc_table_id < 0 ||
            gateup_desc_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) ||
            down_desc_table_id < 0 ||
            down_desc_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillPipeline] invalid descriptor table id");
            return false;
        }

        const auto &gateup_table = grouped_gateup_desc_tables_[gateup_desc_table_id];
        const auto &down_table = grouped_down_desc_tables_[down_desc_table_id];
        if (!gateup_table.valid || !down_table.valid ||
            gateup_table.num_experts != num_experts ||
            down_table.num_experts != num_experts ||
            gateup_table.d_model != d_model ||
            down_table.d_model != d_model ||
            gateup_table.intermediate != intermediate ||
            down_table.intermediate != intermediate)
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillPipeline] descriptor table shape mismatch");
            return false;
        }

        void *stream = requireStream("CUDAMoEKernel::executeGroupedPrefillPipeline");
        const DeviceId device = deviceId();
        const int total_slots = seq_len * top_k;
        const int max_tokens_per_expert = seq_len;
        if (!ensureGroupedPrefillScratchCapacity(total_slots, d_model, intermediate) ||
            !ensureTensorOnDevice(hidden, device, stream, "hidden") ||
            !ensureOutputOnDevice(output, device, stream, "output"))
            return false;

        const float *d_hidden = static_cast<const float *>(hidden->gpu_data_ptr());
        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!d_hidden || !d_output)
            return false;

        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        cudaError_t err = cudaMemsetAsync(d_output, 0,
                                          static_cast<size_t>(seq_len) * d_model * sizeof(float),
                                          cuda_stream);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillPipeline] output memset failed: "
                      << cudaGetErrorString(err));
            return false;
        }

        const bool shared_expert_group = prepared_num_experts_ == 1 && num_experts == 1 && top_k == 1;
        const bool ok = cudaMoE_grouped_prefill_pipeline(
            d_hidden,
            gateup_table.device_gate_descs,
            gateup_table.device_up_descs,
            down_table.device_descs,
            d_group_counts_,
            d_group_offsets_,
            d_group_token_indices_,
            shared_expert_group ? nullptr : d_group_original_to_grouped_,
            d_group_active_expert_ids_,
            d_group_weights_,
            d_prefill_A_int8_,
            d_prefill_A_scales_,
            d_prefill_gate_,
            d_prefill_up_,
            d_prefill_swiglu_int8_,
            d_prefill_swiglu_scales_,
            d_prefill_gate_,
            d_output,
            num_experts,
            d_model,
            intermediate,
            max_tokens_per_expert,
            total_slots,
            top_k,
            group_active_expert_slots_,
            gateup_table.codebook_id,
            down_table.codebook_id,
            device_ordinal_,
            stream);
        if (!ok)
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillPipeline] grouped CUDA pipeline failed");
            return false;
        }

        markDeviceWritten(output, device, stream);
        const int active_expert_slots = group_active_expert_slots_;
        const int selected_tile_m = selectGroupedPrefillTileM(
            debugEnv().gemm.cuda_moe_prefill_tile_m,
            max_tokens_per_expert);
        recordGroupedPrefillCounters(
            seq_len,
            top_k,
            num_experts,
            active_expert_slots,
            selected_tile_m,
            debugEnv().gemm.cuda_moe_prefill_fuse_swiglu);
        return true;
    }

    bool CUDAMoEKernel::groupedExpertGateUpDecodeFromTable(
        const TensorBase *input,
        const int *expert_ids,
        int table_id,
        int num_active,
        ITensor *const *gate_outputs,
        ITensor *const *up_outputs,
        int d_model,
        int intermediate)
    {
        if (!input || !expert_ids || table_id < 0 || num_active <= 0 ||
            !gate_outputs || !up_outputs || d_model <= 0 || intermediate <= 0)
            return false;
        if (num_active > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            (d_model % 32) != 0)
            return false;
        if (table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()))
            return false;

        const auto &table = grouped_gateup_desc_tables_[table_id];
        if (!table.valid || !table.device_gate_descs || !table.device_up_descs ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromTable] descriptor table mismatch");
            return false;
        }
        for (int slot = 0; slot < num_active; ++slot)
        {
            if (expert_ids[slot] < 0 || expert_ids[slot] >= table.num_experts)
                return false;
        }
        if (!ensureGroupedDecodeMetadata(expert_ids, nullptr, num_active, false))
            return false;

        void *stream = requireStream("CUDAMoEKernel::groupedExpertGateUpDecodeFromTable");
        const DeviceId device = deviceId();
        if (!setMoEDevice(device_ordinal_, "groupedExpertGateUpDecodeFromTable"))
            return false;
        if (!requireTensorType(input, TensorType::FP32, "input", "groupedExpertGateUpDecodeFromTable") ||
            !requireMatrixCapacity(input, 1, d_model, "input", "groupedExpertGateUpDecodeFromTable") ||
            !requireTensorElements(input, static_cast<size_t>(d_model), "input", "groupedExpertGateUpDecodeFromTable"))
            return false;

        const int k_partitions = debugEnv().gemm.cuda_moe_gateup_kparts;
        const bool use_kpart = debugEnv().gemm.cuda_moe_gateup_kpart_decode &&
                               ensureGroupedGateUpKPartScratchCapacity(num_active, k_partitions, intermediate);
        if (!ensureGroupedGateUpDecodeCapacity(num_active, d_model))
            return false;

        const float *d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        if (!d_hidden && isGraphCaptureActive())
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromTable] input upload required during graph capture");
            return false;
        }
        if (!d_hidden)
        {
            auto *mutable_input = const_cast<TensorBase *>(input);
            if (!mutable_input->ensureOnDevice(device, stream))
                return false;
            d_hidden = static_cast<const float *>(mutable_input->gpu_data_ptr());
        }
        if (!d_hidden)
            return false;
        const int blocks_per_row = d_model / 32;
        if (!requireCudaDevicePointer(d_hidden, device_ordinal_, "input", "groupedExpertGateUpDecodeFromTable", stream) ||
            !requireCudaDevicePointer(d_decode_hidden_int8_, device_ordinal_, "decode hidden int8", "groupedExpertGateUpDecodeFromTable", stream) ||
            !requireCudaDevicePointer(d_decode_hidden_scales_, device_ordinal_, "decode hidden scales", "groupedExpertGateUpDecodeFromTable", stream))
            return false;
        if (use_kpart &&
            (!requireCudaDevicePointer(d_grouped_gateup_gate_partials_, device_ordinal_, "gate partials", "groupedExpertGateUpDecodeFromTable", stream) ||
             !requireCudaDevicePointer(d_grouped_gateup_up_partials_, device_ordinal_, "up partials", "groupedExpertGateUpDecodeFromTable", stream)))
            return false;
        if (!requireTensorElements(input, static_cast<size_t>(blocks_per_row) * 32u, "input", "groupedExpertGateUpDecodeFromTable"))
            return false;

        std::array<float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        for (int slot = 0; slot < num_active; ++slot)
        {
            if (!gate_outputs[slot] || !up_outputs[slot])
                return false;
            if ((!gate_outputs[slot]->gpu_data_ptr() || !up_outputs[slot]->gpu_data_ptr()) &&
                isGraphCaptureActive())
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromTable] output allocation required during graph capture");
                return false;
            }
            if (!ensureOutputOnDevice(gate_outputs[slot], device, stream, "gate_output") ||
                !ensureOutputOnDevice(up_outputs[slot], device, stream, "up_output"))
                return false;
            auto *gate_output_base = asTensorBase(gate_outputs[slot], "groupedExpertGateUpDecodeFromTable gate_output");
            auto *up_output_base = asTensorBase(up_outputs[slot], "groupedExpertGateUpDecodeFromTable up_output");
            if (!requireTensorType(gate_output_base, TensorType::FP32, "gate_output", "groupedExpertGateUpDecodeFromTable") ||
                !requireTensorType(up_output_base, TensorType::FP32, "up_output", "groupedExpertGateUpDecodeFromTable") ||
                !requireTensorElements(gate_outputs[slot], static_cast<size_t>(intermediate), "gate_output", "groupedExpertGateUpDecodeFromTable") ||
                !requireTensorElements(up_outputs[slot], static_cast<size_t>(intermediate), "up_output", "groupedExpertGateUpDecodeFromTable"))
                return false;
            gate_ptrs[slot] = static_cast<float *>(gate_outputs[slot]->gpu_data_ptr());
            up_ptrs[slot] = static_cast<float *>(up_outputs[slot]->gpu_data_ptr());
            if (!gate_ptrs[slot] || !up_ptrs[slot])
                return false;
            if (!requireAlignedPointer(gate_ptrs[slot], 16, "gate_output", "groupedExpertGateUpDecodeFromTable") ||
                !requireAlignedPointer(up_ptrs[slot], 16, "up_output", "groupedExpertGateUpDecodeFromTable") ||
                !requireCudaDevicePointer(gate_ptrs[slot], device_ordinal_, "gate_output", "groupedExpertGateUpDecodeFromTable", stream) ||
                !requireCudaDevicePointer(up_ptrs[slot], device_ordinal_, "up_output", "groupedExpertGateUpDecodeFromTable", stream))
                return false;
        }

        float **d_gate_ptrs = nullptr;
        float **d_up_ptrs = nullptr;
        if (!ensureRuntimeGateUpPointerArrays(table_id, num_active, gate_ptrs, up_ptrs,
                                              &d_gate_ptrs, &d_up_ptrs))
            return false;

        const bool ok = use_kpart
                            ? cudaMoE_grouped_gate_up_native_vnni_decode_table_kpart(
                                  d_hidden,
                                  table.device_gate_descs,
                                  table.device_up_descs,
                                  d_grouped_decode_expert_ids_,
                                  d_gate_ptrs,
                                  d_up_ptrs,
                                  d_decode_hidden_int8_,
                                  d_decode_hidden_scales_,
                                  d_grouped_gateup_gate_partials_,
                                  d_grouped_gateup_up_partials_,
                                  num_active,
                                  intermediate,
                                  d_model,
                                  table.num_experts,
                                  table.codebook_id,
                                  k_partitions,
                                  device_ordinal_,
                                  stream)
                            : cudaMoE_grouped_gate_up_native_vnni_decode_table(
                                  d_hidden,
                                  table.device_gate_descs,
                                  table.device_up_descs,
                                  d_grouped_decode_expert_ids_,
                                  d_gate_ptrs,
                                  d_up_ptrs,
                                  d_decode_hidden_int8_,
                                  d_decode_hidden_scales_,
                                  num_active,
                                  intermediate,
                                  d_model,
                                  table.codebook_id,
                                  device_ordinal_,
                                  stream);
        if (ok)
        {
            for (int slot = 0; slot < num_active; ++slot)
            {
                markDeviceWritten(gate_outputs[slot], device, stream);
                markDeviceWritten(up_outputs[slot], device, stream);
            }
            recordGroupedDecodeCounter(
                "cuda_moe_grouped_decode_gateup_calls", "table", num_active,
                d_model, intermediate, use_kpart ? "kpart" : "serial");
        }
        return ok;
    }

    bool CUDAMoEKernel::groupedExpertDownDecodeFromTable(
        ITensor *const *gate_tensors,
        ITensor *const *up_tensors,
        const int *expert_ids,
        const float *expert_weights,
        int table_id,
        int num_active,
        ITensor *output,
        int d_model,
        int intermediate)
    {
        if (!gate_tensors || !up_tensors || !expert_ids || !expert_weights ||
            table_id < 0 || num_active <= 0 || !output ||
            d_model <= 0 || intermediate <= 0)
            return false;
        if (num_active > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            (intermediate % 32) != 0)
            return false;
        if (table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
            return false;

        const auto &table = grouped_down_desc_tables_[table_id];
        if (!table.valid || !table.device_descs ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromTable] descriptor table mismatch");
            return false;
        }
        for (int slot = 0; slot < num_active; ++slot)
        {
            if (expert_ids[slot] < 0 || expert_ids[slot] >= table.num_experts)
                return false;
        }
        if (!ensureGroupedDecodeMetadata(expert_ids, expert_weights, num_active, true))
            return false;

        void *stream = requireStream("CUDAMoEKernel::groupedExpertDownDecodeFromTable");
        const DeviceId device = deviceId();
        if (!ensureGroupedDownDecodeCapacity(num_active, intermediate) ||
            !setMoEDevice(device_ordinal_, "groupedExpertDownDecodeFromTable"))
            return false;

        std::array<const float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        for (int slot = 0; slot < num_active; ++slot)
        {
            if (!gate_tensors[slot] || !up_tensors[slot])
                return false;
            gate_ptrs[slot] = static_cast<const float *>(gate_tensors[slot]->gpu_data_ptr());
            up_ptrs[slot] = static_cast<const float *>(up_tensors[slot]->gpu_data_ptr());
            if (!gate_ptrs[slot] || !up_ptrs[slot])
                return false;
        }

        if (!output->gpu_data_ptr() && isGraphCaptureActive())
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromTable] output allocation required during graph capture");
            return false;
        }
        if (!ensureOutputOnDevice(output, device, stream, "moe_output"))
            return false;

        const float **d_gate_ptrs = nullptr;
        const float **d_up_ptrs = nullptr;
        if (!ensureRuntimeDownPointerArrays(table_id, num_active, gate_ptrs, up_ptrs,
                                            &d_gate_ptrs, &d_up_ptrs))
            return false;

        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        const int k_partitions = debugEnv().gemm.cuda_moe_down_kparts;
        const bool use_kpart = debugEnv().gemm.cuda_moe_down_kpart_decode &&
                               ensureGroupedDownKPartScratchCapacity(k_partitions, d_model, num_active);

        const bool ok = use_kpart
            ? cudaMoE_grouped_swiglu_down_native_vnni_decode_table_kpart(
                  d_gate_ptrs,
                  d_up_ptrs,
                  table.device_descs,
                  d_grouped_decode_expert_ids_,
                  d_grouped_decode_weights_,
                  d_decode_swiglu_int8_,
                  d_decode_swiglu_scales_,
                  d_grouped_down_partials_,
                  d_output,
                  num_active,
                  d_model,
                  intermediate,
                  table.num_experts,
                  table.codebook_id,
                  k_partitions,
                  device_ordinal_,
                  stream)
            : cudaMoE_grouped_swiglu_down_native_vnni_decode_table(
                  d_gate_ptrs,
                  d_up_ptrs,
                  table.device_descs,
                  d_grouped_decode_expert_ids_,
                  d_grouped_decode_weights_,
                  d_decode_swiglu_int8_,
                  d_decode_swiglu_scales_,
                  d_output,
                  num_active,
                  d_model,
                  intermediate,
                  table.codebook_id,
                  device_ordinal_,
                  stream);
        if (ok)
        {
            markDeviceWritten(output, device, stream);
            recordGroupedDecodeCounter(
                "cuda_moe_grouped_decode_down_calls", "table", num_active,
                d_model, intermediate, use_kpart ? "kpart" : "serial");
        }
        return ok;
    }

    bool CUDAMoEKernel::groupedExpertDecodeFromRuntime(
        DeviceMoELayerRuntime *runtime_layer,
        const TensorBase *input,
        int gateup_table_id,
        int down_table_id,
        int top_k,
        ITensor *output,
        int d_model,
        int intermediate)
    {
        if (!runtime_layer || !input || gateup_table_id < 0 || down_table_id < 0 ||
            top_k <= 0 || !output || d_model <= 0 || intermediate <= 0)
            return false;
        if (top_k > static_cast<int>(kDeviceMoEMaxTopK) ||
            top_k > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            (d_model % 32) != 0 || (intermediate % 32) != 0)
            return false;
        if (gateup_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) ||
            down_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
            return false;

        const auto &gateup_table = grouped_gateup_desc_tables_[gateup_table_id];
        const auto &down_table = grouped_down_desc_tables_[down_table_id];
        if (!gateup_table.valid || !gateup_table.device_gate_descs || !gateup_table.device_up_descs ||
            !down_table.valid || !down_table.device_descs ||
            gateup_table.d_model != d_model || gateup_table.intermediate != intermediate ||
            down_table.d_model != d_model || down_table.intermediate != intermediate)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDecodeFromRuntime] descriptor table mismatch");
            return false;
        }

        void *stream = requireStream("CUDAMoEKernel::groupedExpertDecodeFromRuntime");
        const DeviceId device = deviceId();
        if (!setMoEDevice(device_ordinal_, "groupedExpertDecodeFromRuntime"))
            return false;

        const int gateup_k_partitions = debugEnv().gemm.cuda_moe_gateup_kparts;
        const bool use_gateup_kpart =
            debugEnv().gemm.cuda_moe_gateup_kpart_decode &&
            ensureGroupedGateUpKPartScratchCapacity(top_k, gateup_k_partitions, intermediate);
        const int down_k_partitions = debugEnv().gemm.cuda_moe_down_kparts;
        const bool use_down_kpart =
            debugEnv().gemm.cuda_moe_down_kpart_decode &&
            ensureGroupedDownKPartScratchCapacity(down_k_partitions, d_model, top_k);

        if (!ensureGroupedPrefillScratchCapacity(top_k, d_model, intermediate) ||
            !ensureGroupedGateUpDecodeCapacity(top_k, d_model) ||
            !ensureGroupedDownDecodeCapacity(top_k, intermediate))
            return false;

        const float *d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        if (!d_hidden && isGraphCaptureActive())
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDecodeFromRuntime] input upload required during graph capture");
            return false;
        }
        if (!d_hidden)
        {
            auto *mutable_input = const_cast<TensorBase *>(input);
            if (!mutable_input->ensureOnDevice(device, stream))
                return false;
            d_hidden = static_cast<const float *>(mutable_input->gpu_data_ptr());
        }
        if (!d_hidden || !d_prefill_gate_ || !d_prefill_up_)
            return false;

        if (!output->gpu_data_ptr() && isGraphCaptureActive())
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDecodeFromRuntime] output allocation required during graph capture");
            return false;
        }
        if (!ensureOutputOnDevice(output, device, stream, "moe_output"))
            return false;

        const int max_dim = std::max(d_model, intermediate);
        std::array<float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK> const_gate_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK> const_up_ptrs = {};
        for (int slot = 0; slot < top_k; ++slot)
        {
            gate_ptrs[slot] = d_prefill_gate_ + static_cast<size_t>(slot) * max_dim;
            up_ptrs[slot] = d_prefill_up_ + static_cast<size_t>(slot) * intermediate;
            const_gate_ptrs[slot] = gate_ptrs[slot];
            const_up_ptrs[slot] = up_ptrs[slot];
        }

        float **d_gate_ptrs = nullptr;
        float **d_up_ptrs = nullptr;
        if (!ensureRuntimeGateUpPointerArrays(gateup_table_id, top_k, gate_ptrs, up_ptrs,
                                              &d_gate_ptrs, &d_up_ptrs))
            return false;
        const float **d_down_gate_ptrs = nullptr;
        const float **d_down_up_ptrs = nullptr;
        if (!ensureRuntimeDownPointerArrays(down_table_id, top_k, const_gate_ptrs, const_up_ptrs,
                                            &d_down_gate_ptrs, &d_down_up_ptrs))
            return false;

        const int *d_expert_ids = runtimeTopKExpertIdsDevice(runtime_layer);
        const float *d_weights = runtimeTopKWeightsDevice(runtime_layer);
        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!d_expert_ids || !d_weights || !d_output)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDecodeFromRuntime] missing runtime/output device pointer");
            return false;
        }

        const bool gateup_ok = use_gateup_kpart
                                   ? cudaMoE_grouped_gate_up_native_vnni_decode_table_kpart(
                                         d_hidden,
                                         gateup_table.device_gate_descs,
                                         gateup_table.device_up_descs,
                                         d_expert_ids,
                                         d_gate_ptrs,
                                         d_up_ptrs,
                                         d_decode_hidden_int8_,
                                         d_decode_hidden_scales_,
                                         d_grouped_gateup_gate_partials_,
                                         d_grouped_gateup_up_partials_,
                                         top_k,
                                         intermediate,
                                         d_model,
                                         gateup_table.num_experts,
                                         gateup_table.codebook_id,
                                         gateup_k_partitions,
                                         device_ordinal_,
                                         stream)
                                   : cudaMoE_grouped_gate_up_native_vnni_decode_table(
                                         d_hidden,
                                         gateup_table.device_gate_descs,
                                         gateup_table.device_up_descs,
                                         d_expert_ids,
                                         d_gate_ptrs,
                                         d_up_ptrs,
                                         d_decode_hidden_int8_,
                                         d_decode_hidden_scales_,
                                         top_k,
                                         intermediate,
                                         d_model,
                                         gateup_table.codebook_id,
                                         device_ordinal_,
                                         stream);
        if (!gateup_ok)
            return false;

        const bool down_ok = use_down_kpart
                                 ? cudaMoE_grouped_swiglu_down_native_vnni_decode_table_kpart(
                                       d_down_gate_ptrs,
                                       d_down_up_ptrs,
                                       down_table.device_descs,
                                       d_expert_ids,
                                       d_weights,
                                       d_decode_swiglu_int8_,
                                       d_decode_swiglu_scales_,
                                       d_grouped_down_partials_,
                                       d_output,
                                       top_k,
                                       d_model,
                                       intermediate,
                                       down_table.num_experts,
                                       down_table.codebook_id,
                                       down_k_partitions,
                                       device_ordinal_,
                                       stream)
                                 : cudaMoE_grouped_swiglu_down_native_vnni_decode_table(
                                       d_down_gate_ptrs,
                                       d_down_up_ptrs,
                                       down_table.device_descs,
                                       d_expert_ids,
                                       d_weights,
                                       d_decode_swiglu_int8_,
                                       d_decode_swiglu_scales_,
                                       d_output,
                                       top_k,
                                       d_model,
                                       intermediate,
                                       down_table.codebook_id,
                                       device_ordinal_,
                                       stream);
        if (!down_ok)
            return false;

        markDeviceWritten(output, device, stream);
        recordGroupedDecodeCounter(
            "cuda_moe_grouped_decode_fused_calls", "runtime", top_k,
            d_model, intermediate, "fused_block_down");
        if (!isGraphCaptureActive() && !isCudaStreamCapturing(stream))
        {
            recordFusedDecodeTimer("cuda_moe_fused_decode_hidden_quantize", top_k, d_model, intermediate);
            recordFusedDecodeTimer("cuda_moe_fused_decode_gateup_kpart", top_k, d_model, intermediate);
            recordFusedDecodeTimer("cuda_moe_fused_decode_swiglu_quantize", top_k, d_model, intermediate);
            recordFusedDecodeTimer("cuda_moe_fused_decode_down_warp_reduce", top_k, d_model, intermediate);
        }
        return true;
    }

    bool CUDAMoEKernel::groupedExpertGateUpDecodeFromRuntime(
        DeviceMoELayerRuntime *runtime_layer,
        const TensorBase *input,
        int table_id,
        int top_k,
        ITensor *const *gate_outputs,
        ITensor *const *up_outputs,
        int d_model,
        int intermediate)
    {
        if (!runtime_layer || !input || table_id < 0 || top_k <= 0 ||
            !gate_outputs || !up_outputs || d_model <= 0 || intermediate <= 0)
            return false;
        if (top_k > static_cast<int>(kDeviceMoEMaxTopK) ||
            top_k > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            (d_model % 32) != 0)
            return false;
        if (table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()))
            return false;

        const auto &table = grouped_gateup_desc_tables_[table_id];
        if (!table.valid || !table.device_gate_descs || !table.device_up_descs ||
            table.num_experts <= 0 || table.d_model != d_model ||
            table.intermediate != intermediate)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRuntime] descriptor table mismatch");
            return false;
        }

        void *stream = requireStream("CUDAMoEKernel::groupedExpertGateUpDecodeFromRuntime");
        const DeviceId device = deviceId();
        if (!ensureGroupedGateUpDecodeCapacity(top_k, d_model))
            return false;

        // Decide whether to use the split-K (kpart) decode path. It requires
        // pre-sized partials scratch; if the scratch cannot be ensured (e.g. the
        // shape grew during graph capture), fall back to the serial kernel.
        const int k_partitions = debugEnv().gemm.cuda_moe_gateup_kparts;
        const bool use_kpart = debugEnv().gemm.cuda_moe_gateup_kpart_decode &&
                               ensureGroupedGateUpKPartScratchCapacity(top_k, k_partitions, intermediate);

        if (!setMoEDevice(device_ordinal_, "groupedExpertGateUpDecodeFromRuntime"))
            return false;

        const float *d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        if (!d_hidden)
        {
            if (isGraphCaptureActive())
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRuntime] input upload required during graph capture");
                return false;
            }
            if (!const_cast<TensorBase *>(input)->ensureOnDevice(device, stream))
                return false;
            d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        }

        std::array<float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        for (int slot = 0; slot < top_k; ++slot)
        {
            if (!gate_outputs[slot] || !up_outputs[slot])
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRuntime] null output tensor for slot "
                          << slot);
                return false;
            }
            if ((!gate_outputs[slot]->gpu_data_ptr() || !up_outputs[slot]->gpu_data_ptr()) &&
                isGraphCaptureActive())
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRuntime] output allocation required during graph capture");
                return false;
            }
            if (!ensureOutputOnDevice(gate_outputs[slot], device, stream, "gate_output") ||
                !ensureOutputOnDevice(up_outputs[slot], device, stream, "up_output"))
                return false;

            gate_ptrs[slot] = static_cast<float *>(gate_outputs[slot]->gpu_data_ptr());
            up_ptrs[slot] = static_cast<float *>(up_outputs[slot]->gpu_data_ptr());
            if (!gate_ptrs[slot] || !up_ptrs[slot])
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRuntime] missing output device pointer for slot "
                          << slot);
                return false;
            }
        }

        float **d_gate_ptrs = nullptr;
        float **d_up_ptrs = nullptr;
        if (!ensureRuntimeGateUpPointerArrays(table_id, top_k, gate_ptrs, up_ptrs,
                                              &d_gate_ptrs, &d_up_ptrs))
            return false;

        const int *d_expert_ids = runtimeTopKExpertIdsDevice(runtime_layer);
        if (!d_hidden || !d_expert_ids)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRuntime] missing runtime/input device pointer");
            return false;
        }

        const bool ok = use_kpart
                            ? cudaMoE_grouped_gate_up_native_vnni_decode_table_kpart(
                                  d_hidden,
                                  table.device_gate_descs,
                                  table.device_up_descs,
                                  d_expert_ids,
                                  d_gate_ptrs,
                                  d_up_ptrs,
                                  d_decode_hidden_int8_,
                                  d_decode_hidden_scales_,
                                  d_grouped_gateup_gate_partials_,
                                  d_grouped_gateup_up_partials_,
                                  top_k,
                                  intermediate,
                                  d_model,
                                  table.num_experts,
                                  table.codebook_id,
                                  k_partitions,
                                  device_ordinal_,
                                  stream)
                            : cudaMoE_grouped_gate_up_native_vnni_decode_table(
                                  d_hidden,
                                  table.device_gate_descs,
                                  table.device_up_descs,
                                  d_expert_ids,
                                  d_gate_ptrs,
                                  d_up_ptrs,
                                  d_decode_hidden_int8_,
                                  d_decode_hidden_scales_,
                                  top_k,
                                  intermediate,
                                  d_model,
                                  table.codebook_id,
                                  device_ordinal_,
                                  stream);

        if (ok)
        {
            for (int slot = 0; slot < top_k; ++slot)
            {
                markDeviceWritten(gate_outputs[slot], device, stream);
                markDeviceWritten(up_outputs[slot], device, stream);
            }
            recordGroupedDecodeCounter(
                "cuda_moe_grouped_decode_gateup_calls", "runtime", top_k,
                d_model, intermediate, use_kpart ? "kpart" : "serial");
        }
        return ok;
    }

    bool CUDAMoEKernel::groupedExpertDownDecodeFromRuntime(
        ITensor *const *gate_tensors,
        ITensor *const *up_tensors,
        DeviceMoELayerRuntime *runtime_layer,
        int table_id,
        int top_k,
        ITensor *output,
        int d_model,
        int intermediate)
    {
        if (!gate_tensors || !up_tensors || !runtime_layer || !output ||
            table_id < 0 || top_k <= 0 || d_model <= 0 || intermediate <= 0)
            return false;
        if (top_k > static_cast<int>(kDeviceMoEMaxTopK) ||
            top_k > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            (intermediate % 32) != 0)
            return false;
        if (table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
            return false;

        const auto &table = grouped_down_desc_tables_[table_id];
        if (!table.valid || !table.device_descs || table.num_experts <= 0 ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRuntime] descriptor table mismatch");
            return false;
        }

        void *stream = requireStream("CUDAMoEKernel::groupedExpertDownDecodeFromRuntime");
        const DeviceId device = deviceId();
        if (!ensureGroupedDownDecodeCapacity(top_k, intermediate))
            return false;
        if (!setMoEDevice(device_ordinal_, "groupedExpertDownDecodeFromRuntime"))
            return false;

        std::array<const float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        for (int slot = 0; slot < top_k; ++slot)
        {
            if (!gate_tensors[slot] || !up_tensors[slot])
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRuntime] null gate/up tensor for slot "
                          << slot);
                return false;
            }
            gate_ptrs[slot] = static_cast<const float *>(gate_tensors[slot]->gpu_data_ptr());
            up_ptrs[slot] = static_cast<const float *>(up_tensors[slot]->gpu_data_ptr());
            if (!gate_ptrs[slot] || !up_ptrs[slot])
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRuntime] missing gate/up device pointer for slot "
                          << slot);
                return false;
            }
        }

        if (!output->gpu_data_ptr() && isGraphCaptureActive())
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRuntime] output allocation required during graph capture");
            return false;
        }
        if (!ensureOutputOnDevice(output, device, stream, "moe_output"))
            return false;

        const float **d_gate_ptrs = nullptr;
        const float **d_up_ptrs = nullptr;
        if (!ensureRuntimeDownPointerArrays(table_id, top_k, gate_ptrs, up_ptrs,
                                            &d_gate_ptrs, &d_up_ptrs))
            return false;

        const int *d_expert_ids = runtimeTopKExpertIdsDevice(runtime_layer);
        const float *d_weights = runtimeTopKWeightsDevice(runtime_layer);
        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!d_expert_ids || !d_weights || !d_output)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRuntime] missing runtime/output device pointer");
            return false;
        }

        // Split-K (K-partition) path raises occupancy by multiplying the block
        // count by k_partitions; fall back to the serial full-K path when the
        // toggle is off or the partial scratch cannot be sized.
        const int k_partitions = debugEnv().gemm.cuda_moe_down_kparts;
        const bool use_kpart = debugEnv().gemm.cuda_moe_down_kpart_decode &&
                               ensureGroupedDownKPartScratchCapacity(k_partitions, d_model, top_k);

        const bool ok = use_kpart
            ? cudaMoE_grouped_swiglu_down_native_vnni_decode_table_kpart(
                  d_gate_ptrs,
                  d_up_ptrs,
                  table.device_descs,
                  d_expert_ids,
                  d_weights,
                  d_decode_swiglu_int8_,
                  d_decode_swiglu_scales_,
                  d_grouped_down_partials_,
                  d_output,
                  top_k,
                  d_model,
                  intermediate,
                  table.num_experts,
                  table.codebook_id,
                  k_partitions,
                  device_ordinal_,
                  stream)
            : cudaMoE_grouped_swiglu_down_native_vnni_decode_table(
                  d_gate_ptrs,
                  d_up_ptrs,
                  table.device_descs,
                  d_expert_ids,
                  d_weights,
                  d_decode_swiglu_int8_,
                  d_decode_swiglu_scales_,
                  d_output,
                  top_k,
                  d_model,
                  intermediate,
                  table.codebook_id,
                  device_ordinal_,
                  stream);

        if (ok)
        {
            markDeviceWritten(output, device, stream);
            recordGroupedDecodeCounter(
                "cuda_moe_grouped_decode_down_calls", "runtime", top_k,
                d_model, intermediate, use_kpart ? "kpart" : "serial");
        }
        return ok;
    }

    int CUDAMoEKernel::getExpertTokenCount(int expert_id) const
    {
        if (expert_id < 0 || expert_id >= prepared_num_experts_)
            return 0;
        return host_expert_counts_[expert_id];
    }

    void CUDAMoEKernel::gatherExpertBatch(ITensor *hidden, ITensor *batch_buffer,
                                          int expert_id, int d_model)
    {
        const int count = getExpertTokenCount(expert_id);
        if (count <= 0)
            return;
        void *stream = requireStream("CUDAMoEKernel::gatherExpertBatch");
        const DeviceId device = deviceId();
        if (!ensureTensorOnDevice(hidden, device, stream, "hidden") ||
            !ensureOutputOnDevice(batch_buffer, device, stream, "batch_buffer"))
            return;
        const int offset = host_expert_offsets_[expert_id];
        gatherTokenBatch(static_cast<const float *>(hidden->gpu_data_ptr()),
                         static_cast<float *>(batch_buffer->gpu_data_ptr()),
                         d_group_token_indices_ + offset, count, d_model);
        markDeviceWritten(batch_buffer, device, stream);
    }

    void CUDAMoEKernel::scatterExpertResults(ITensor *output, ITensor *expert_results,
                                             int expert_id, int d_model)
    {
        const int count = getExpertTokenCount(expert_id);
        if (count <= 0)
            return;
        void *stream = requireStream("CUDAMoEKernel::scatterExpertResults");
        const DeviceId device = deviceId();
        if (!ensureTensorOnDevice(output, device, stream, "output") ||
            !ensureTensorOnDevice(expert_results, device, stream, "expert_results"))
            return;
        const int offset = host_expert_offsets_[expert_id];
        scatterAddWeighted(static_cast<float *>(output->gpu_data_ptr()),
                           static_cast<const float *>(expert_results->gpu_data_ptr()),
                           d_group_token_indices_ + offset,
                           d_group_weights_ + offset,
                           count, d_model);
        markDeviceWritten(output, device, stream);
    }

} // namespace llaminar2
