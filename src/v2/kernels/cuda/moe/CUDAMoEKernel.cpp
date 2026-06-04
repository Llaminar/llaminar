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
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../utils/Logger.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/PerfStatsCollector.h"

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

    void recordGroupedDecodeCounter(
        const std::string &name,
        const char *source,
        const char *route,
        llaminar2::DeviceId device,
        int active_slots,
        int d_model,
        int intermediate,
        int codebook_id,
        int k_partitions)
    {
        if (!llaminar2::PerfStatsCollector::isEnabled())
            return;

        llaminar2::PerfStatsCollector::addCounter(
            "kernel",
            name,
            1.0,
            "moe",
            device.to_string(),
            llaminar2::PerfStatsCollector::Tags{
                {"source", source},
                {"route", route},
                {"active_slots", std::to_string(active_slots)},
                {"d_model", std::to_string(d_model)},
                {"intermediate", std::to_string(intermediate)},
                {"codebook", std::to_string(codebook_id)},
                {"k_partitions", std::to_string(k_partitions)}});
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

    bool cudaMoE_route_logits_bf16_gate(
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
        const int *routing_indices, const float *routing_weights,
        const int *expert_offsets, const int *expert_counts,
        int *grouped_token_indices, float *grouped_weights,
        int total_slots, int top_k, int num_experts,
        int device_idx, void *stream);

    bool cudaMoE_group_tokens_small_float(
        const float *routing_indices, const float *routing_weights,
        int *expert_counts, int *expert_offsets,
        int *grouped_token_indices, float *grouped_weights,
        int *active_expert_ids,
        int total_slots, int num_experts, int top_k,
        int max_active_experts,
        int device_idx, void *stream);

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
        const float *d_group_weights,
        const int *d_active_expert_ids,
        int8_t *d_scratch_A_int8,
        float *d_scratch_scales,
        float *d_scratch_gate,
        float *d_scratch_up,
        int8_t *d_scratch_swiglu_int8,
        float *d_scratch_swiglu_scales,
        float *d_scratch_down_out,
        float *d_gate_partials,
        float *d_up_partials,
        float *d_output,
        int num_experts,
        int d_model,
        int intermediate,
        int max_tokens_per_expert,
        int total_slots,
        int active_expert_slots,
        uint8_t gateup_codebook_id,
        uint8_t down_codebook_id,
        int gateup_k_partitions,
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

        release(d_staging_indices_);
        release(d_staging_weights_);
        release(d_route_logits_);
        release(d_route_indices_);
        release(d_route_weights_);
        release(d_group_int_indices_);
        release(d_group_offsets_);
        release(d_group_counts_);
        release(d_group_token_indices_);
        release(d_group_write_heads_);
        release(d_group_weights_);
        release(d_group_active_expert_ids_);
        for (auto &table : grouped_down_desc_tables_)
            release(table.device_descs);
        for (auto &table : grouped_gateup_desc_tables_)
        {
            release(table.device_gate_descs);
            release(table.device_up_descs);
        }
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
        grouped_down_desc_tables_.clear();
        grouped_gateup_desc_tables_.clear();
        runtime_gateup_pointer_cache_.clear();
        runtime_down_pointer_cache_.clear();
    }

    bool CUDAMoEKernel::ensureStagingCapacity(int count)
    {
        if (count <= staging_capacity_)
            return true;
        if (!setMoEDevice(device_ordinal_, "ensureStagingCapacity"))
            return false;

        if (d_staging_indices_)
            cudaFree(d_staging_indices_);
        if (d_staging_weights_)
            cudaFree(d_staging_weights_);
        d_staging_indices_ = nullptr;
        d_staging_weights_ = nullptr;

        cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_staging_indices_), static_cast<size_t>(count) * sizeof(int));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel] cudaMalloc staging indices failed: " << cudaGetErrorString(err));
            staging_capacity_ = 0;
            return false;
        }
        err = cudaMalloc(reinterpret_cast<void **>(&d_staging_weights_), static_cast<size_t>(count) * sizeof(float));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel] cudaMalloc staging weights failed: " << cudaGetErrorString(err));
            cudaFree(d_staging_indices_);
            d_staging_indices_ = nullptr;
            staging_capacity_ = 0;
            return false;
        }

        staging_capacity_ = count;
        return true;
    }

    bool CUDAMoEKernel::ensureRouteBufferCapacity(size_t logits_count, size_t topk_count)
    {
        if (!setMoEDevice(device_ordinal_, "ensureRouteBufferCapacity"))
            return false;

        if (logits_count > route_logits_capacity_)
        {
            if (d_route_logits_)
                cudaFree(d_route_logits_);
            d_route_logits_ = nullptr;
            cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_route_logits_), logits_count * sizeof(float));
            if (err != cudaSuccess)
            {
                LOG_ERROR("[CUDAMoEKernel] cudaMalloc route logits failed: " << cudaGetErrorString(err));
                route_logits_capacity_ = 0;
                return false;
            }
            route_logits_capacity_ = logits_count;
        }

        if (topk_count > route_topk_capacity_)
        {
            if (d_route_indices_)
                cudaFree(d_route_indices_);
            if (d_route_weights_)
                cudaFree(d_route_weights_);
            d_route_indices_ = nullptr;
            d_route_weights_ = nullptr;

            cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_route_indices_), topk_count * sizeof(int));
            if (err != cudaSuccess)
            {
                LOG_ERROR("[CUDAMoEKernel] cudaMalloc route indices failed: " << cudaGetErrorString(err));
                route_topk_capacity_ = 0;
                return false;
            }
            err = cudaMalloc(reinterpret_cast<void **>(&d_route_weights_), topk_count * sizeof(float));
            if (err != cudaSuccess)
            {
                LOG_ERROR("[CUDAMoEKernel] cudaMalloc route weights failed: " << cudaGetErrorString(err));
                cudaFree(d_route_indices_);
                d_route_indices_ = nullptr;
                route_topk_capacity_ = 0;
                return false;
            }
            route_topk_capacity_ = topk_count;
        }

        return true;
    }

    bool CUDAMoEKernel::ensureGroupingBufferCapacity(int total_slots, int num_experts)
    {
        if (!setMoEDevice(device_ordinal_, "ensureGroupingBufferCapacity"))
            return false;

        if (total_slots > group_slots_cap_)
        {
            if (isGraphCaptureActive())
            {
                LOG_ERROR("[CUDAMoEKernel] grouping buffer realloc during graph capture (need "
                          << total_slots << " slots, have " << group_slots_cap_ << ")");
                return false;
            }
            if (d_group_int_indices_)
                cudaFree(d_group_int_indices_);
            if (d_group_token_indices_)
                cudaFree(d_group_token_indices_);
            if (d_group_weights_)
                cudaFree(d_group_weights_);
            if (d_group_active_expert_ids_)
                cudaFree(d_group_active_expert_ids_);
            d_group_int_indices_ = nullptr;
            d_group_token_indices_ = nullptr;
            d_group_weights_ = nullptr;
            d_group_active_expert_ids_ = nullptr;

            cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_group_int_indices_), static_cast<size_t>(total_slots) * sizeof(int));
            if (err != cudaSuccess)
                return false;
            err = cudaMalloc(reinterpret_cast<void **>(&d_group_token_indices_), static_cast<size_t>(total_slots) * sizeof(int));
            if (err != cudaSuccess)
                return false;
            err = cudaMalloc(reinterpret_cast<void **>(&d_group_weights_), static_cast<size_t>(total_slots) * sizeof(float));
            if (err != cudaSuccess)
                return false;
            err = cudaMalloc(reinterpret_cast<void **>(&d_group_active_expert_ids_), static_cast<size_t>(total_slots) * sizeof(int));
            if (err != cudaSuccess)
                return false;
            group_slots_cap_ = total_slots;
        }

        if (num_experts > group_experts_cap_)
        {
            if (isGraphCaptureActive())
            {
                LOG_ERROR("[CUDAMoEKernel] expert grouping buffer realloc during graph capture (need "
                          << num_experts << " experts, have " << group_experts_cap_ << ")");
                return false;
            }
            if (d_group_offsets_)
                cudaFree(d_group_offsets_);
            if (d_group_counts_)
                cudaFree(d_group_counts_);
            if (d_group_write_heads_)
                cudaFree(d_group_write_heads_);
            d_group_offsets_ = nullptr;
            d_group_counts_ = nullptr;
            d_group_write_heads_ = nullptr;

            cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_group_offsets_), static_cast<size_t>(num_experts) * sizeof(int));
            if (err != cudaSuccess)
                return false;
            err = cudaMalloc(reinterpret_cast<void **>(&d_group_counts_), static_cast<size_t>(num_experts) * sizeof(int));
            if (err != cudaSuccess)
                return false;
            err = cudaMalloc(reinterpret_cast<void **>(&d_group_write_heads_), static_cast<size_t>(num_experts) * sizeof(int));
            if (err != cudaSuccess)
                return false;
            group_experts_cap_ = num_experts;
        }

        return true;
    }

    bool CUDAMoEKernel::ensureGroupedPrefillScratchCapacity(int total_slots, int d_model, int intermediate)
    {
        const bool need_realloc = total_slots > prefill_slots_cap_ ||
                                  d_model > prefill_d_model_cap_ ||
                                  intermediate > prefill_intermediate_cap_;
        if (!need_realloc)
            return true;

        if (isGraphCaptureActive())
        {
            LOG_ERROR("[CUDAMoEKernel] grouped prefill scratch realloc during graph capture (need slots="
                      << total_slots << " d_model=" << d_model << " intermediate=" << intermediate
                      << ", have slots=" << prefill_slots_cap_ << " d_model=" << prefill_d_model_cap_
                      << " intermediate=" << prefill_intermediate_cap_ << ")");
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "ensureGroupedPrefillScratchCapacity"))
            return false;

        auto release = [](auto *&ptr)
        {
            if (ptr)
            {
                cudaFree(ptr);
                ptr = nullptr;
            }
        };
        release(d_prefill_A_int8_);
        release(d_prefill_A_scales_);
        release(d_prefill_swiglu_int8_);
        release(d_prefill_swiglu_scales_);
        release(d_prefill_gate_);
        release(d_prefill_up_);

        const int max_dim = std::max(d_model, intermediate);
        const int max_blocks = (max_dim + 31) / 32;
        cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_prefill_A_int8_),
                                     static_cast<size_t>(total_slots) * max_dim * sizeof(int8_t));
        if (err == cudaSuccess)
            err = cudaMalloc(reinterpret_cast<void **>(&d_prefill_A_scales_),
                             static_cast<size_t>(total_slots) * max_blocks * sizeof(float));
        if (err == cudaSuccess)
            err = cudaMalloc(reinterpret_cast<void **>(&d_prefill_swiglu_int8_),
                             static_cast<size_t>(total_slots) * intermediate * sizeof(int8_t));
        if (err == cudaSuccess)
            err = cudaMalloc(reinterpret_cast<void **>(&d_prefill_swiglu_scales_),
                             static_cast<size_t>(total_slots) * ((intermediate + 31) / 32) * sizeof(float));
        if (err == cudaSuccess)
            err = cudaMalloc(reinterpret_cast<void **>(&d_prefill_gate_),
                             static_cast<size_t>(total_slots) * max_dim * sizeof(float));
        if (err == cudaSuccess)
            err = cudaMalloc(reinterpret_cast<void **>(&d_prefill_up_),
                             static_cast<size_t>(total_slots) * intermediate * sizeof(float));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel] grouped prefill scratch cudaMalloc failed: " << cudaGetErrorString(err));
            release(d_prefill_A_int8_);
            release(d_prefill_A_scales_);
            release(d_prefill_swiglu_int8_);
            release(d_prefill_swiglu_scales_);
            release(d_prefill_gate_);
            release(d_prefill_up_);
            prefill_slots_cap_ = 0;
            prefill_d_model_cap_ = 0;
            prefill_intermediate_cap_ = 0;
            return false;
        }

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

        if (isGraphCaptureActive())
        {
            LOG_ERROR("[CUDAMoEKernel] grouped gate/up decode scratch growth during graph capture (need top_k="
                      << top_k << " d_model=" << d_model << ", have top_k=" << decode_gateup_topk_cap_
                      << " d_model=" << decode_gateup_d_model_cap_ << ")");
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "ensureGroupedGateUpDecodeCapacity"))
            return false;

        auto release = [](auto *&ptr)
        {
            if (ptr)
            {
                cudaFree(ptr);
                ptr = nullptr;
            }
        };
        release(d_decode_hidden_int8_);
        release(d_decode_hidden_scales_);

        const int blocks_per_row = d_model / 32;
        cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_decode_hidden_int8_),
                                     static_cast<size_t>(d_model) * sizeof(int8_t));
        if (err == cudaSuccess)
            err = cudaMalloc(reinterpret_cast<void **>(&d_decode_hidden_scales_),
                             static_cast<size_t>(blocks_per_row) * sizeof(float));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel] grouped gate/up decode scratch cudaMalloc failed: "
                      << cudaGetErrorString(err));
            release(d_decode_hidden_int8_);
            release(d_decode_hidden_scales_);
            decode_gateup_topk_cap_ = 0;
            decode_gateup_d_model_cap_ = 0;
            return false;
        }

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

        // Growth requires fresh allocations, which is illegal mid-capture.
        if (isGraphCaptureActive())
        {
            LOG_ERROR("[CUDAMoEKernel] grouped gate/up kpart scratch growth during graph capture (need top_k="
                      << top_k << " k_partitions=" << k_partitions << " intermediate=" << intermediate << ")");
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "ensureGroupedGateUpKPartScratchCapacity"))
            return false;

        auto release = [](auto *&ptr)
        {
            if (ptr)
            {
                cudaFree(ptr);
                ptr = nullptr;
            }
        };
        release(d_grouped_gateup_gate_partials_);
        release(d_grouped_gateup_up_partials_);

        const size_t partial_count = static_cast<size_t>(top_k) *
                                     static_cast<size_t>(k_partitions) *
                                     static_cast<size_t>(intermediate);
        cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_grouped_gateup_gate_partials_),
                                     partial_count * sizeof(float));
        if (err == cudaSuccess)
            err = cudaMalloc(reinterpret_cast<void **>(&d_grouped_gateup_up_partials_),
                             partial_count * sizeof(float));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel] grouped gate/up kpart scratch cudaMalloc failed: "
                      << cudaGetErrorString(err));
            release(d_grouped_gateup_gate_partials_);
            release(d_grouped_gateup_up_partials_);
            grouped_gateup_kpart_active_cap_ = 0;
            grouped_gateup_kpart_partitions_cap_ = 0;
            grouped_gateup_kpart_intermediate_cap_ = 0;
            return false;
        }

        grouped_gateup_kpart_active_cap_ = top_k;
        grouped_gateup_kpart_partitions_cap_ = k_partitions;
        grouped_gateup_kpart_intermediate_cap_ = intermediate;
        return true;
    }

    bool CUDAMoEKernel::ensureGroupedDownKPartScratchCapacity(int k_partitions, int d_model)
    {
        // Only the discrete partition counts the kpart launcher accepts are valid.
        if (d_model <= 0 ||
            !(k_partitions == 2 || k_partitions == 4 || k_partitions == 8 ||
              k_partitions == 16))
            return false;

        // Fast path: the existing partial buffer already covers the requested shape.
        if (d_grouped_down_partials_ &&
            grouped_down_kpart_partitions_cap_ >= k_partitions &&
            grouped_down_kpart_d_model_cap_ >= d_model)
            return true;

        // Growth requires a fresh allocation, which is illegal mid-capture.
        if (isGraphCaptureActive())
        {
            LOG_ERROR("[CUDAMoEKernel] grouped down kpart scratch growth during graph capture (need k_partitions="
                      << k_partitions << " d_model=" << d_model << ")");
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "ensureGroupedDownKPartScratchCapacity"))
            return false;

        auto release = [](auto *&ptr)
        {
            if (ptr)
            {
                cudaFree(ptr);
                ptr = nullptr;
            }
        };
        release(d_grouped_down_partials_);

        // The down partials buffer is [k_partitions][d_model] floats (the output
        // dimension is d_model, not intermediate).
        const size_t partial_count =
            static_cast<size_t>(k_partitions) * static_cast<size_t>(d_model);
        cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_grouped_down_partials_),
                                     partial_count * sizeof(float));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel] grouped down kpart scratch cudaMalloc failed: "
                      << cudaGetErrorString(err));
            release(d_grouped_down_partials_);
            grouped_down_kpart_partitions_cap_ = 0;
            grouped_down_kpart_d_model_cap_ = 0;
            return false;
        }

        grouped_down_kpart_partitions_cap_ = k_partitions;
        grouped_down_kpart_d_model_cap_ = d_model;
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

        if (isGraphCaptureActive())
        {
            LOG_ERROR("[CUDAMoEKernel] grouped down decode scratch growth during graph capture (need top_k="
                      << top_k << " intermediate=" << intermediate << ", have top_k=" << decode_down_topk_cap_
                      << " intermediate=" << decode_down_intermediate_cap_ << ")");
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "ensureGroupedDownDecodeCapacity"))
            return false;

        auto release = [](auto *&ptr)
        {
            if (ptr)
            {
                cudaFree(ptr);
                ptr = nullptr;
            }
        };
        release(d_decode_swiglu_int8_);
        release(d_decode_swiglu_scales_);

        const int blocks_per_row = intermediate / 32;
        cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_decode_swiglu_int8_),
                                     static_cast<size_t>(top_k) * intermediate * sizeof(int8_t));
        if (err == cudaSuccess)
            err = cudaMalloc(reinterpret_cast<void **>(&d_decode_swiglu_scales_),
                             static_cast<size_t>(top_k) * blocks_per_row * sizeof(float));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel] grouped down decode scratch cudaMalloc failed: "
                      << cudaGetErrorString(err));
            release(d_decode_swiglu_int8_);
            release(d_decode_swiglu_scales_);
            decode_down_topk_cap_ = 0;
            decode_down_intermediate_cap_ = 0;
            return false;
        }

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
            return false;

        auto cached_matches = [&]()
        {
            if (static_cast<int>(grouped_decode_cached_expert_ids_.size()) != num_active)
                return false;
            for (int i = 0; i < num_active; ++i)
            {
                if (grouped_decode_cached_expert_ids_[i] != expert_ids[i])
                    return false;
            }
            if (include_weights)
            {
                if (static_cast<int>(grouped_decode_cached_weights_.size()) != num_active)
                    return false;
                for (int i = 0; i < num_active; ++i)
                {
                    if (grouped_decode_cached_weights_[i] != expert_weights[i])
                        return false;
                }
            }
            return true;
        };

        if (num_active <= grouped_decode_metadata_cap_ &&
            d_grouped_decode_expert_ids_ &&
            (!include_weights || d_grouped_decode_weights_) &&
            cached_matches())
        {
            return true;
        }

        if (isGraphCaptureActive())
        {
            LOG_ERROR("[CUDAMoEKernel] grouped decode metadata cache miss during graph capture");
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "ensureGroupedDecodeMetadata"))
            return false;

        if (num_active > grouped_decode_metadata_cap_)
        {
            if (d_grouped_decode_expert_ids_)
                cudaFree(d_grouped_decode_expert_ids_);
            if (d_grouped_decode_weights_)
                cudaFree(d_grouped_decode_weights_);
            d_grouped_decode_expert_ids_ = nullptr;
            d_grouped_decode_weights_ = nullptr;

            cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_grouped_decode_expert_ids_),
                                         static_cast<size_t>(num_active) * sizeof(int));
            if (err == cudaSuccess)
                err = cudaMalloc(reinterpret_cast<void **>(&d_grouped_decode_weights_),
                                 static_cast<size_t>(num_active) * sizeof(float));
            if (err != cudaSuccess)
            {
                LOG_ERROR("[CUDAMoEKernel] grouped decode metadata cudaMalloc failed: "
                          << cudaGetErrorString(err));
                if (d_grouped_decode_expert_ids_)
                    cudaFree(d_grouped_decode_expert_ids_);
                if (d_grouped_decode_weights_)
                    cudaFree(d_grouped_decode_weights_);
                d_grouped_decode_expert_ids_ = nullptr;
                d_grouped_decode_weights_ = nullptr;
                grouped_decode_metadata_cap_ = 0;
                grouped_decode_cached_expert_ids_.clear();
                grouped_decode_cached_weights_.clear();
                return false;
            }
            grouped_decode_metadata_cap_ = num_active;
        }

        cudaStream_t stream = static_cast<cudaStream_t>(getStream());
        cudaError_t err = cudaMemcpyAsync(
            d_grouped_decode_expert_ids_,
            expert_ids,
            static_cast<size_t>(num_active) * sizeof(int),
            cudaMemcpyHostToDevice,
            stream);
        if (err == cudaSuccess && include_weights)
        {
            err = cudaMemcpyAsync(
                d_grouped_decode_weights_,
                expert_weights,
                static_cast<size_t>(num_active) * sizeof(float),
                cudaMemcpyHostToDevice,
                stream);
        }
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel] grouped decode metadata upload failed: "
                      << cudaGetErrorString(err));
            return false;
        }

        grouped_decode_cached_expert_ids_.assign(expert_ids, expert_ids + num_active);
        if (include_weights)
            grouped_decode_cached_weights_.assign(expert_weights, expert_weights + num_active);
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

        switch (gate_type)
        {
        case TensorType::FP32:
            if (!cudaMoE_route_logits(hidden, static_cast<const float *>(gate_weights), d_route_logits_,
                                      seq_len, d_model, num_experts,
                                      device_ordinal_, getStream()))
                return false;
            break;
        case TensorType::BF16:
            if (!cudaMoE_route_logits_bf16_gate(hidden, gate_weights, d_route_logits_,
                                                seq_len, d_model, num_experts,
                                                device_ordinal_, getStream()))
                return false;
            break;
        default:
            LOG_ERROR("[CUDAMoEKernel] Unsupported MoE router gate tensor type "
                      << tensorTypeName(gate_type) << " for CUDA routing");
            return false;
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

        if (hidden->native_type() != TensorType::FP32)
        {
            LOG_ERROR("[CUDAMoEKernel::routeWithTensors] hidden must be FP32, got "
                      << tensorTypeName(hidden->native_type()));
            return false;
        }
        if (output_indices->native_type() != TensorType::FP32 ||
            output_weights->native_type() != TensorType::FP32)
        {
            LOG_ERROR("[CUDAMoEKernel::routeWithTensors] routing outputs must be FP32 tensors");
            return false;
        }

        const float *d_hidden = static_cast<const float *>(hidden->gpu_data_ptr());
        const void *d_gate = gate_weights->gpu_data_ptr();
        float *d_idx = static_cast<float *>(output_indices->gpu_data_ptr());
        float *d_wt = static_cast<float *>(output_weights->gpu_data_ptr());
        if (!d_hidden || !d_gate || !d_idx || !d_wt)
            return false;

        DeviceRouteBuffers buffers;
        if (!routeCore(d_hidden, d_gate, gate_weights->native_type(),
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
        const bool needs_decode_host_topk = (seq_len == 1);
#ifdef ENABLE_PIPELINE_SNAPSHOTS
        const bool needs_snapshot_host_topk = true;
#else
        const bool needs_snapshot_host_topk = false;
#endif
        const bool needs_host_topk = needs_decode_host_topk || needs_snapshot_host_topk;
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
        host_result.router_logits.resize(buffers.logits_count);
        err = cudaMemcpyAsync(host_result.router_logits.data(), buffers.d_logits,
                              buffers.logits_count * sizeof(float),
                              cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream));
        if (err != cudaSuccess)
            return false;
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
        if (hidden->native_type() != TensorType::FP32)
        {
            LOG_ERROR("[CUDAMoEKernel::decodeRouteSelect] hidden must be FP32, got "
                      << tensorTypeName(hidden->native_type()));
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
            if (output_indices->native_type() != TensorType::FP32 ||
                output_weights->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[CUDAMoEKernel::decodeRouteSelect] legacy routing outputs must be FP32 tensors");
                return false;
            }
            legacy_indices = static_cast<float *>(output_indices->gpu_data_ptr());
            legacy_weights = static_cast<float *>(output_weights->gpu_data_ptr());
        }

        const float *d_hidden = static_cast<const float *>(hidden->gpu_data_ptr());
        const void *d_gate = gate_weights->gpu_data_ptr();
        if (!d_hidden || !d_gate)
            return false;

        if (!ensureRouteBufferCapacity(static_cast<size_t>(num_experts), static_cast<size_t>(top_k)))
            return false;
        switch (gate_weights->native_type())
        {
        case TensorType::FP32:
            if (!cudaMoE_route_logits(d_hidden, static_cast<const float *>(d_gate), d_route_logits_,
                                      /*seq_len=*/1, d_model, num_experts,
                                      device_ordinal_, stream))
                return false;
            break;
        case TensorType::BF16:
            if (!cudaMoE_route_logits_bf16_gate(d_hidden, d_gate, d_route_logits_,
                                                /*seq_len=*/1, d_model, num_experts,
                                                device_ordinal_, stream))
                return false;
            break;
        default:
            LOG_ERROR("[CUDAMoEKernel] Unsupported MoE router gate tensor type "
                      << tensorTypeName(gate_weights->native_type()) << " for CUDA decode routing");
            return false;
        }
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
               cudaMoE_scatter_tokens_deterministic(d_routing_indices, d_routing_weights,
                                                    d_expert_offsets, d_expert_counts,
                                                    d_grouped_token_indices, d_grouped_weights,
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
        group_active_expert_slots_ = 0;
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

        const bool use_small_grouping =
            total_slots <= 64 &&
            num_experts <= 256 &&
            top_k <= 16;
        if (use_small_grouping)
        {
            const int active_expert_slots = std::min(total_slots, num_experts);
            if (!cudaMoE_group_tokens_small_float(
                    d_float_indices,
                    d_float_weights,
                    d_group_counts_,
                    d_group_offsets_,
                    d_group_token_indices_,
                    d_group_weights_,
                    d_group_active_expert_ids_,
                    total_slots,
                    num_experts,
                    top_k,
                    active_expert_slots,
                    device_ordinal_,
                    stream))
            {
                LOG_ERROR("[CUDAMoEKernel::prepareExpertGroupsAsync] small-M grouping failed");
                group_active_expert_slots_ = 0;
                return false;
            }

            group_active_expert_slots_ = active_expert_slots;
            prepared_num_experts_ = num_experts;
            if (PerfStatsCollector::isEnabled())
            {
                PerfStatsCollector::addCounter(
                    "kernel",
                    "cuda_moe_small_prefill_grouping_calls",
                    1.0,
                    "moe",
                    device.to_string(),
                    PerfStatsCollector::Tags{
                        {"total_slots", std::to_string(total_slots)},
                        {"num_experts", std::to_string(num_experts)},
                        {"top_k", std::to_string(top_k)}});
            }
            return true;
        }

        group_active_expert_slots_ = 0;
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
        const int active_expert_slots = group_active_expert_slots_;
        if (active_expert_slots > 0 && !d_group_active_expert_ids_)
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillPipeline] compact active expert list missing");
            return false;
        }
        if (!ensureGroupedPrefillScratchCapacity(total_slots, d_model, intermediate) ||
            !ensureTensorOnDevice(hidden, device, stream, "hidden") ||
            !ensureOutputOnDevice(output, device, stream, "output"))
            return false;
        const int selected_tile_m = debugEnv().gemm.cuda_moe_prefill_tile_m == 0
                                        ? (seq_len <= 2 ? 2 : (seq_len <= 4 ? 4 : 16))
                                        : debugEnv().gemm.cuda_moe_prefill_tile_m;
        const int selected_tile_n = selected_tile_m <= 2 ? 64 : 128;
        const int gateup_k_partitions = debugEnv().gemm.cuda_moe_gateup_kparts;
        const bool use_gateup_kpart_swiglu =
            debugEnv().gemm.cuda_moe_gateup_kpart_decode &&
            debugEnv().gemm.cuda_moe_prefill_fuse_swiglu &&
            active_expert_slots > 0 &&
            selected_tile_m <= 4 &&
            seq_len <= 4;
        if (use_gateup_kpart_swiglu &&
            !ensureGroupedGateUpKPartScratchCapacity(total_slots, gateup_k_partitions, intermediate))
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillPipeline] verifier small-M split-K gate/up scratch unavailable");
            return false;
        }

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

        const bool ok = cudaMoE_grouped_prefill_pipeline(
            d_hidden,
            gateup_table.device_gate_descs,
            gateup_table.device_up_descs,
            down_table.device_descs,
            d_group_counts_,
            d_group_offsets_,
            d_group_token_indices_,
            d_group_weights_,
            d_group_active_expert_ids_,
            d_prefill_A_int8_,
            d_prefill_A_scales_,
            d_prefill_gate_,
            d_prefill_up_,
            // The fused gate/up epilogue reads A while writing SwiGLU quant output,
            // so SwiGLU scratch must not alias d_prefill_A_*.
            d_prefill_swiglu_int8_,
            d_prefill_swiglu_scales_,
            d_prefill_gate_,
            use_gateup_kpart_swiglu ? d_grouped_gateup_gate_partials_ : nullptr,
            use_gateup_kpart_swiglu ? d_grouped_gateup_up_partials_ : nullptr,
            d_output,
            num_experts,
            d_model,
            intermediate,
            max_tokens_per_expert,
            total_slots,
            active_expert_slots,
            gateup_table.codebook_id,
            down_table.codebook_id,
            gateup_k_partitions,
            device_ordinal_,
            stream);
        if (!ok)
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillPipeline] grouped CUDA pipeline failed");
            return false;
        }

        markDeviceWritten(output, device, stream);
        if (PerfStatsCollector::isEnabled())
        {
            PerfStatsCollector::addCounter(
                "kernel",
                "cuda_moe_grouped_prefill_swiglu_path_calls",
                1.0,
                "moe",
                device.to_string(),
                PerfStatsCollector::Tags{
                    {"swiglu_path", debugEnv().gemm.cuda_moe_prefill_fuse_swiglu ? "fused" : "split"},
                    {"total_slots", std::to_string(total_slots)},
                    {"active_expert_slots", std::to_string(active_expert_slots)},
                    {"num_experts", std::to_string(num_experts)},
                    {"tile_m", std::to_string(selected_tile_m)},
                    {"tile_n", std::to_string(selected_tile_n)},
                    {"gateup_route", use_gateup_kpart_swiglu ? "kpart_swiglu" : "serial"}});
        }
        if (PerfStatsCollector::isEnabled() && active_expert_slots > 0)
        {
            PerfStatsCollector::addCounter(
                "kernel",
                "cuda_moe_grouped_prefill_active_expert_grid_calls",
                1.0,
                "moe",
                device.to_string(),
                PerfStatsCollector::Tags{
                    {"total_slots", std::to_string(total_slots)},
                    {"active_expert_slots", std::to_string(active_expert_slots)},
                    {"num_experts", std::to_string(num_experts)},
                    {"tile_m", std::to_string(selected_tile_m)},
                    {"tile_n", std::to_string(selected_tile_n)},
                    {"swiglu_path", debugEnv().gemm.cuda_moe_prefill_fuse_swiglu ? "fused" : "split"},
                    {"gateup_route", use_gateup_kpart_swiglu ? "kpart_swiglu" : "serial"}});
        }
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
        if (num_active > static_cast<int>(kDeviceMoEMaxTopK) ||
            num_active > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            (d_model % 32) != 0)
            return false;
        if (table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()))
            return false;

        const auto &table = grouped_gateup_desc_tables_[table_id];
        if (!table.valid || !table.device_gate_descs || !table.device_up_descs ||
            table.num_experts <= 0 || table.d_model != d_model ||
            table.intermediate != intermediate)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromTable] descriptor table mismatch");
            return false;
        }
        for (int slot = 0; slot < num_active; ++slot)
        {
            if (expert_ids[slot] < 0 || expert_ids[slot] >= table.num_experts)
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromTable] invalid expert id "
                          << expert_ids[slot] << " at slot " << slot);
                return false;
            }
        }

        void *stream = requireStream("CUDAMoEKernel::groupedExpertGateUpDecodeFromTable");
        const DeviceId device = deviceId();
        if (!ensureGroupedGateUpDecodeCapacity(num_active, d_model) ||
            !ensureGroupedDecodeMetadata(expert_ids, nullptr, num_active, false))
            return false;

        const int k_partitions = debugEnv().gemm.cuda_moe_gateup_kparts;
        const bool use_kpart = debugEnv().gemm.cuda_moe_gateup_kpart_decode &&
                               ensureGroupedGateUpKPartScratchCapacity(num_active, k_partitions, intermediate);

        if (!setMoEDevice(device_ordinal_, "groupedExpertGateUpDecodeFromTable"))
            return false;

        const float *d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        if (!d_hidden)
        {
            if (isGraphCaptureActive())
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromTable] input upload required during graph capture");
                return false;
            }
            if (!const_cast<TensorBase *>(input)->ensureOnDevice(device, stream))
                return false;
            d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        }

        std::array<float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        for (int slot = 0; slot < num_active; ++slot)
        {
            if (!gate_outputs[slot] || !up_outputs[slot])
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromTable] null output tensor for slot "
                          << slot);
                return false;
            }
            if ((!gate_outputs[slot]->gpu_data_ptr() || !up_outputs[slot]->gpu_data_ptr()) &&
                isGraphCaptureActive())
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromTable] output allocation required during graph capture");
                return false;
            }
            if (!ensureOutputOnDevice(gate_outputs[slot], device, stream, "gate_output") ||
                !ensureOutputOnDevice(up_outputs[slot], device, stream, "up_output"))
                return false;

            gate_ptrs[slot] = static_cast<float *>(gate_outputs[slot]->gpu_data_ptr());
            up_ptrs[slot] = static_cast<float *>(up_outputs[slot]->gpu_data_ptr());
            if (!gate_ptrs[slot] || !up_ptrs[slot])
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromTable] missing output device pointer for slot "
                          << slot);
                return false;
            }
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
                "cuda_moe_grouped_decode_gateup_calls",
                "table",
                use_kpart ? "kpart" : "serial",
                device,
                num_active,
                d_model,
                intermediate,
                table.codebook_id,
                use_kpart ? k_partitions : 1);
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
        if (!gate_tensors || !up_tensors || !expert_ids || !expert_weights || !output ||
            table_id < 0 || num_active <= 0 || d_model <= 0 || intermediate <= 0)
            return false;
        if (num_active > static_cast<int>(kDeviceMoEMaxTopK) ||
            num_active > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            (intermediate % 32) != 0)
            return false;
        if (table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
            return false;

        const auto &table = grouped_down_desc_tables_[table_id];
        if (!table.valid || !table.device_descs || table.num_experts <= 0 ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromTable] descriptor table mismatch");
            return false;
        }
        for (int slot = 0; slot < num_active; ++slot)
        {
            if (expert_ids[slot] < 0 || expert_ids[slot] >= table.num_experts)
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromTable] invalid expert id "
                          << expert_ids[slot] << " at slot " << slot);
                return false;
            }
        }

        void *stream = requireStream("CUDAMoEKernel::groupedExpertDownDecodeFromTable");
        const DeviceId device = deviceId();
        if (!ensureGroupedDownDecodeCapacity(num_active, intermediate) ||
            !ensureGroupedDecodeMetadata(expert_ids, expert_weights, num_active, true))
            return false;
        if (!setMoEDevice(device_ordinal_, "groupedExpertDownDecodeFromTable"))
            return false;

        std::array<const float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        for (int slot = 0; slot < num_active; ++slot)
        {
            if (!gate_tensors[slot] || !up_tensors[slot])
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromTable] null gate/up tensor for slot "
                          << slot);
                return false;
            }
            gate_ptrs[slot] = static_cast<const float *>(gate_tensors[slot]->gpu_data_ptr());
            up_ptrs[slot] = static_cast<const float *>(up_tensors[slot]->gpu_data_ptr());
            if (!gate_ptrs[slot] || !up_ptrs[slot])
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromTable] missing gate/up device pointer for slot "
                          << slot);
                return false;
            }
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

        const int k_partitions = debugEnv().gemm.cuda_moe_down_kparts;
        const bool use_kpart = debugEnv().gemm.cuda_moe_down_kpart_decode &&
                               ensureGroupedDownKPartScratchCapacity(k_partitions, d_model);

        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!d_output)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromTable] missing output device pointer");
            return false;
        }

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
                "cuda_moe_grouped_decode_down_calls",
                "table",
                use_kpart ? "kpart" : "serial",
                device,
                num_active,
                d_model,
                intermediate,
                table.codebook_id,
                use_kpart ? k_partitions : 1);
        }
        return ok;
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
                "cuda_moe_grouped_decode_gateup_calls",
                "runtime",
                use_kpart ? "kpart" : "serial",
                device,
                top_k,
                d_model,
                intermediate,
                table.codebook_id,
                use_kpart ? k_partitions : 1);
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
                               ensureGroupedDownKPartScratchCapacity(k_partitions, d_model);

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
                "cuda_moe_grouped_decode_down_calls",
                "runtime",
                use_kpart ? "kpart" : "serial",
                device,
                top_k,
                d_model,
                intermediate,
                table.codebook_id,
                use_kpart ? k_partitions : 1);
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
