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

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
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

    /**
     * @brief True when MoE prefill grouping wrapper tracing is requested.
     *
     * This intentionally uses the same environment flag as the stage-level
     * trace.  The wrapper log is a lower observation point for captured prefill
     * graphs where the ordinary stage body may not be visible during replay.
     */
    bool traceMoEPrefillGroupingWrapperEnabled()
    {
        const bool assignment_trace =
            !llaminar2::DebugEnv::isFalseyEnv("LLAMINAR_MOE_PREFILL_ASSIGNMENT_TRACE");
        return assignment_trace || llaminar2::debugEnv().execution.prefill_graph_trace;
    }

    /**
     * @brief Add bytes to a deterministic FNV-1a diagnostic hash.
     */
    uint64_t updateMoETraceHash(uint64_t hash, const void *data, size_t bytes)
    {
        constexpr uint64_t kFnvPrime = 1099511628211ULL;
        const auto *raw = static_cast<const uint8_t *>(data);
        for (size_t i = 0; i < bytes; ++i)
        {
            hash ^= static_cast<uint64_t>(raw[i]);
            hash *= kFnvPrime;
        }
        return hash;
    }

    /**
     * @brief Hash a contiguous vector for compact prefill grouping diagnostics.
     */
    template <typename T>
    uint64_t hashMoETraceVector(const std::vector<T> &values)
    {
        constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
        if (values.empty())
            return kFnvOffset;
        return updateMoETraceHash(
            kFnvOffset,
            values.data(),
            values.size() * sizeof(T));
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
        // Snapshot-enabled activation tensors may be zero-copy mapped host memory.
        // They are valid CUDA kernel inputs only when the pointer is the
        // device-visible alias returned by cudaHostGetDevicePointer(), not an
        // arbitrary host pointer.
        const bool mapped_device_alias =
            attr.type == cudaMemoryTypeHost &&
            attr.device == expected_device &&
            attr.devicePointer == ptr;
        const bool type_ok = attr.type == cudaMemoryTypeDevice ||
                             attr.type == cudaMemoryTypeManaged ||
                             mapped_device_alias;
        if (type_ok && attr.device == expected_device)
            return true;
        LOG_ERROR("[CUDAMoEKernel] " << context << " requires " << name
                                      << " pointer on CUDA device " << expected_device
                                      << ", got attr.device=" << attr.device
                                      << " attr.type=" << static_cast<int>(attr.type)
                                      << " device_ptr=" << attr.devicePointer
                                      << " host_ptr=" << attr.hostPointer
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

        // MTP verifier prefill runs with M=1..4 rows and must be cheap enough
        // to be the production grouped path.  The compact TM=2 template wins on
        // CUDA for these tiny active groups because it avoids over-wide blocks
        // while still covering the full row set through grid.y.
        if (max_tokens_per_expert <= 4)
            return 2;
        if (max_tokens_per_expert <= 8)
            return 8;
        return 16;
    }

    llaminar2::PerfStatsCollector::Tags groupedPrefillTags(
        int seq_len,
        int top_k,
        int num_experts,
        int active_expert_slots,
        int tile_m,
        int tile_n)
    {
        const int total_slots = seq_len * top_k;
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
        int tile_n,
        bool use_gateup_kpart,
        bool fuse_swiglu_requested,
        bool use_ordered_down_kpart,
        bool ordered_scatter)
    {
        auto tags = groupedPrefillTags(seq_len, top_k, num_experts, active_expert_slots, tile_m, tile_n);
        if (active_expert_slots > 0)
        {
            llaminar2::PerfStatsCollector::addCounter(
                "kernel", "cuda_moe_grouped_prefill_active_expert_grid_calls",
                1.0, {}, {}, tags);
        }

        tags["swiglu_path"] =
            (use_gateup_kpart || fuse_swiglu_requested) ? "fused" : "split";
        tags["requested_fused_swiglu"] = fuse_swiglu_requested ? "true" : "false";
        if (active_expert_slots > 0)
        {
            if (use_gateup_kpart)
                tags["gateup_route"] = "kpart_prefill";
            else
                tags["gateup_route"] = fuse_swiglu_requested
                                           ? "fullk_fused_swiglu_prefill"
                                           : "fullk_prefill";

            tags["down_route"] = use_ordered_down_kpart
                                     ? "ordered_kpart_prefill"
                                     : "grouped_prefill";
            tags["down_accumulation"] = use_ordered_down_kpart
                                            ? "row_ordered_kpart"
                                            : (ordered_scatter ? "row_ordered"
                                                               : "slot_scatter");
        }
        else if (ordered_scatter)
        {
            tags["gateup_route"] = "serial";
            tags["down_route"] = "serial";
            tags["down_accumulation"] = "row_ordered";
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

    /**
     * Return true whenever CUDA work is being recorded for replay.
     *
     * Llaminar has an internal graph-capture guard, but several low-level tests
     * and capture utilities use CUDA stream capture directly. Runtime MoE
     * pointer tables must treat both modes identically: all host-staged pointer
     * arrays have to be uploaded before capture begins so replay never depends
     * on stack-owned host arrays or implicit H2D copies.
     */
    bool isCudaMoEDecodeCaptureActive(void *stream)
    {
        return llaminar2::isGraphCaptureActive() || isCudaStreamCapturing(stream);
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

    constexpr uint8_t kCudaMoEMixedCodebookSentinel = 0xffu;

    uint32_t cudaGroupedPrefillCodebookBit(uint8_t cb)
    {
        return cb < 32 ? (uint32_t{1} << cb) : 0u;
    }

    bool cudaGroupedPrefillMaskNeedsIQTables(uint32_t mask)
    {
        for (uint8_t cb = 11; cb <= 17; ++cb)
        {
            if (mask & cudaGroupedPrefillCodebookBit(cb))
                return true;
        }
        return false;
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

    bool validateCudaGroupedDescShape(
        const llaminar2::DeviceNativeVNNIMatrixDesc &desc,
        int n,
        int k)
    {
        return desc.valid() &&
               desc.n == n &&
               desc.k == k &&
               desc.blocks_per_row == static_cast<uint32_t>(k / 32) &&
               cudaGroupedPrefillSupportsCodebook(desc.codebook_id) &&
               (!cudaGroupedPrefillRequiresMins(desc.codebook_id) || desc.mins != nullptr) &&
               (!cudaGroupedPrefillRequiresEmins(desc.codebook_id) || desc.emins != nullptr);
    }

    bool isBlankGroupedDesc(const llaminar2::DeviceNativeVNNIMatrixDesc &desc)
    {
        return desc.payload == nullptr &&
               desc.scales == nullptr &&
               desc.mins == nullptr &&
               desc.emins == nullptr &&
               desc.n == 0 &&
               desc.k == 0 &&
               desc.blocks_per_row == 0;
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

    bool cudaMoE_quantize_router_gate_q8(
        const float *gate_weights, int8_t *gate_weights_q8, float *gate_scales,
        int d_model, int num_experts,
        int device_idx, void *stream);

    bool cudaMoE_gate_logits_single_token_q8_weights(
        const float *hidden, int8_t *hidden_q8, float *hidden_scales,
        const int8_t *gate_weights_q8, const float *gate_scales, float *logits,
        int d_model, int num_experts,
        int device_idx, void *stream);

    bool cudaMoE_gate_logits_q8_weights_decode_equivalent_rows(
        const float *hidden, int8_t *hidden_q8, float *hidden_scales,
        const int8_t *gate_weights_q8, const float *gate_scales, float *logits,
        int seq_len, int d_model, int num_experts,
        int device_idx, void *stream);

    bool cudaMoE_route_logits_decode_equivalent_rows(
        const float *hidden, const float *gate_weights, float *logits,
        int seq_len, int d_model, int num_experts,
        int device_idx, void *stream);

    bool cudaMoE_route_logits_bf16_decode_equivalent_rows(
        const float *hidden, const void *gate_weights, float *logits,
        int seq_len, int d_model, int num_experts,
        int device_idx, void *stream);

    bool cudaMoE_softmax_topk(
        float *logits, int *expert_indices, float *expert_weights,
        int seq_len, int num_experts, int top_k, bool normalize_weights,
        int device_idx, void *stream,
        const int *device_effective_seq_len);

    bool cudaMoE_softmax_topk_decode_runtime(
        float *logits, void *runtime_layer,
        float *legacy_indices, float *legacy_weights,
        int num_experts, int top_k, bool normalize_weights,
        bool write_legacy_outputs, bool update_runtime_histogram,
        void *runtime_layers,
        const void *rebalance_plan_entries,
        uint32_t rebalance_plan_capacity,
        void *rebalance_command_header,
        const void *rebalance_local_transfer_slots,
        uint32_t rebalance_local_transfer_slot_count,
        const void *rebalance_config,
        void *rebalance_apply_status,
        void *rebalance_controller_state,
        int rebalance_target_layer,
        uint32_t rebalance_command_buffer_count,
        int device_idx, void *stream);

    bool cudaMoE_decode_route_select_runtime(
        const int *expert_indices, const float *expert_weights,
        void *runtime_layer,
        float *legacy_indices, float *legacy_weights,
        int num_experts, int top_k, bool write_legacy_outputs,
        bool update_runtime_histogram, int device_idx, void *stream);

    bool cudaMoE_device_rebalance_controller(
        void *runtime_layers,
        const unsigned long long *gathered_histograms,
        void *status,
        const void *config,
        void *plan_entries,
        uint32_t *plan_count,
        uint32_t plan_capacity,
        uint32_t payload_slot_capacity,
        void *command_header,
        void *wave_state,
        void *controller_state,
        uint32_t command_buffer_count,
        int device_idx,
        void *stream);

    bool cudaMoE_pack_rebalance_histograms(
        void *runtime_layers,
        unsigned long long *local_histograms,
        const void *config,
        const void *wave_state,
        const void *controller_state,
        uint32_t command_buffer_count,
        int device_idx,
        void *stream);

    bool cudaMoE_pack_rebalance_directory(
        void *runtime_layers,
        void *local_directory,
        const void *config,
        int device_idx,
        void *stream);

    bool cudaMoE_pack_rebalance_source_descriptors(
        void *runtime_layers,
        const void *gathered_plan_entries,
        const void *gathered_command_headers,
        uint32_t plan_capacity,
        void *local_source_descriptors,
        const void *config,
        void *controller_state,
        uint32_t command_buffer_count,
        int device_idx,
        void *stream);

    bool cudaMoE_project_rebalance_domain_commands(
        const void *gathered_plan_entries,
        const void *gathered_command_headers,
        uint32_t plan_capacity,
        void *local_plan_entries,
        void *local_command_headers,
        const void *config,
        void *status,
        uint32_t payload_slot_capacity,
        uint32_t command_buffer_count,
        const void *gathered_wave_states,
        void *local_wave_states,
        int device_idx,
        void *stream);

    bool cudaMoE_project_prefill_llep_domain_commands(
        const void *gathered_plan_entries,
        const void *gathered_command_headers,
        uint32_t plan_capacity,
        void *local_plan_entries,
        uint32_t *local_plan_count,
        void *local_command_header,
        const void *config,
        void *status,
        uint32_t payload_slot_capacity,
        uint32_t command_buffer_count,
        int device_idx,
        void *stream);

    bool cudaMoE_materialize_prefill_llep_transfer_commands(
        const void *runtime_layer,
        void *plan_entries,
        uint32_t *plan_count,
        uint32_t plan_capacity,
        void *command_header,
        void *status,
        const void *config,
        uint32_t payload_slot_capacity,
        uint32_t layer_idx,
        uint32_t command_buffer_count,
        int device_idx,
        void *stream);

    bool cudaMoE_pack_rebalance_compact_payloads(
        const void *plan_entries,
        const void *command_headers,
        uint32_t plan_capacity,
        const void *local_source_descriptors,
        void *local_payload,
        uint32_t local_payload_slot_count,
        unsigned long long payload_slot_bytes,
        const void *config,
        void *status,
        void *controller_state,
        uint32_t command_buffer_count,
        int device_idx,
        void *stream);

    bool cudaMoE_pack_rebalance_collective_payloads(
        const void *gathered_plan_entries,
        const void *gathered_command_headers,
        uint32_t plan_capacity,
        const void *local_directory,
        void *local_payload,
        uint32_t local_payload_slot_count,
        unsigned long long payload_slot_bytes,
        const void *config,
        void *status,
        void *controller_state,
        uint32_t command_buffer_count,
        int device_idx,
        void *stream);

    bool cudaMoE_unpack_rebalance_collective_payloads(
        const void *plan_entries,
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        const void *command_header,
        const void *gathered_payload,
        uint32_t local_payload_slot_count,
        unsigned long long payload_slot_bytes,
        void *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const void *config,
        void *status,
        void *controller_state,
        uint32_t command_buffer_count,
        int device_idx,
        void *stream);

    bool cudaMoE_init_rebalance_graph_controller_state(
        void *controller_state,
        const void *config,
        int device_idx,
        void *stream);

    bool cudaMoE_publish_rebalance_transfer_complete(
        void *controller_state,
        const void *command_header,
        const void *wave_state,
        const void *copy_status,
        const void *plan_entries,
        uint32_t plan_capacity,
        const void *gathered_copy_status,
        const void *config,
        uint32_t command_buffer_count,
        int device_idx,
        void *stream);

    bool cudaMoE_apply_ready_rebalance_wave(
        void *runtime_layers,
        const void *plan_entries,
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        void *command_header,
        const void *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const void *config,
        void *status,
        void *controller_state,
        int target_layer,
        uint32_t command_buffer_count,
        int device_idx,
        void *stream);

    bool cudaMoE_apply_rebalance_arrivals(
        void *runtime_layers,
        const void *plan_entries,
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        const void *command_header,
        const void *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const void *config,
        void *status,
        int target_layer,
        int device_idx,
        void *stream);

    bool cudaMoE_int_to_float(const int *input, float *output, int count, int device_idx, void *stream);
    bool cudaMoE_float_to_int(const float *input, int *output, int count, int device_idx, void *stream);
    bool cudaMoE_float_to_masked_int(
        const float *input,
        int *output,
        const uint8_t *expert_mask,
        int count,
        int num_experts,
        int device_idx,
        void *stream);

    bool cudaMoE_gather_tokens(
        const float *hidden, float *batch_buffer, const int *token_indices,
        int num_tokens, int d_model, int device_idx, void *stream);

    bool cudaMoE_copy_token_row(
        const float *source, float *row_buffer,
        int row_index, int row_width, int device_idx, void *stream);

    bool cudaMoE_scatter_add(
        float *output, const float *expert_output, const int *token_indices,
        const float *weights, int num_tokens, int d_model, int device_idx, void *stream);

    bool cudaMoE_write_token_row(
        float *destination, const float *row_buffer,
        int row_index, int row_width, int device_idx, void *stream);

    bool cudaMoE_shared_expert_gate(
        const float *input, const float *gate_inp, float *shared_output,
        int seq_len, int d_model, int device_idx, void *stream);

    bool cudaMoE_shared_expert_gate_effective_seq_len(
        const float *input, const float *gate_inp, float *shared_output,
        int seq_len, int d_model, const int *device_effective_seq_len,
        int device_idx, void *stream);

    bool cudaMoE_shared_expert_gate_add(
        const float *input, const float *gate_inp, float *shared_output,
        const float *routed_residual, float *combined_output,
        int seq_len, int d_model, int device_idx, void *stream);

    bool cudaMoE_shared_expert_gate_add_effective_seq_len(
        const float *input, const float *gate_inp, float *shared_output,
        const float *routed_residual, float *combined_output,
        int seq_len, int d_model, const int *device_effective_seq_len,
        int device_idx, void *stream);

    bool cudaMoE_swiglu(float *gate, const float *up, int count, int device_idx, void *stream);
    bool cudaMoE_weighted_add(float *output, const float *input, float weight, int count, int device_idx, void *stream);

    bool cudaMoE_count_per_expert(
        const int *routing_indices, int *expert_counts, int total_slots,
        int num_experts, int device_idx, void *stream);

    bool cudaMoE_exclusive_scan(
        const int *expert_counts, int *expert_offsets,
        int num_experts, int device_idx, void *stream);

    bool cudaMoE_build_active_expert_list(
        const int *expert_counts, int *active_expert_ids,
        int num_experts, int max_active_experts,
        int device_idx, void *stream);

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
        int *original_to_grouped,
        int *original_expert_ids,
        float *grouped_weights,
        int *active_expert_ids,
        int seq_len,
        int device_idx,
        void *stream);

    bool cudaMoE_group_prefill_routes_runtime(
        const float *routing_indices,
        const float *routing_weights,
        void *runtime,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int filter_to_local_runtime_experts,
        int device_idx,
        void *stream);

    bool cudaMoE_regroup_prefill_routes_runtime_assignments(
        void *runtime,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int device_idx,
        void *stream);

    bool cudaMoE_assign_prefill_routes_least_loaded_resident(
        void *runtime,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int device_idx,
        void *stream);

    bool cudaMoE_plan_prefill_routes_least_loaded_current_batch(
        void *runtime,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        uint32_t min_chunk_tokens,
        uint32_t alpha_numerator,
        uint32_t alpha_denominator,
        uint32_t lambda_numerator,
        uint32_t lambda_denominator,
        uint64_t min_spread_improvement,
        uint32_t min_spread_improvement_divisor,
        uint64_t min_spread_improvement_per_transfer,
        uint64_t min_foreign_rows_per_transfer,
        uint32_t max_weight_transfers,
        int enable_balanced_skip,
        int device_idx,
        void *stream);

    bool cudaMoE_assign_prefill_routes_from_llep_current_batch_plan_no_transfers(
        void *runtime,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int device_idx,
        void *stream);

    bool cudaMoE_assign_prefill_routes_from_llep_current_batch_plan_after_transfers(
        void *runtime,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        const void *transfer_status,
        const void *apply_status,
        int device_idx,
        void *stream);

    bool cudaMoE_materialize_runtime_prefill_descriptor_tables(
        const void *runtime,
        llaminar2::DeviceNativeVNNIMatrixDesc *gate_descs,
        llaminar2::DeviceNativeVNNIMatrixDesc *up_descs,
        llaminar2::DeviceNativeVNNIMatrixDesc *down_descs,
        int num_experts,
        int device_idx,
        void *stream);

    bool cudaMoE_build_active_expert_list_runtime(
        const void *runtime,
        int *active_expert_ids,
        int num_experts,
        int max_active_experts,
        int device_idx,
        void *stream);

    bool cudaMoE_build_runtime_original_to_grouped(
        const void *runtime,
        int *original_to_grouped,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int device_idx,
        void *stream);

    bool cudaMoE_prefill_gather_expert_runtime(
        const void *runtime,
        const float *hidden,
        float *batch_buffer,
        int expert_id,
        int max_tokens,
        int d_model,
        int device_idx,
        void *stream);

    bool cudaMoE_prefill_scatter_expert_runtime(
        float *output,
        const float *expert_output,
        const void *runtime,
        int expert_id,
        int max_tokens,
        int d_model,
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
        bool hidden_prequantized,
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
        bool hidden_prequantized,
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

    bool cudaMoE_grouped_gate_up_native_vnni_decode_runtime(
        const float *d_hidden,
        const void *d_runtime_layer,
        const int *d_expert_ids,
        float *const *d_gate_outputs,
        float *const *d_up_outputs,
        int8_t *d_hidden_int8,
        float *d_hidden_scales,
        bool hidden_prequantized,
        int num_active,
        int intermediate,
        int d_model,
        int num_experts,
        uint8_t codebook_id,
        int device_idx,
        void *stream);

    bool cudaMoE_grouped_gate_up_native_vnni_decode_runtime_kpart(
        const float *d_hidden,
        const void *d_runtime_layer,
        const int *d_expert_ids,
        float *const *d_gate_outputs,
        float *const *d_up_outputs,
        int8_t *d_hidden_int8,
        float *d_hidden_scales,
        bool hidden_prequantized,
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

    bool cudaMoE_grouped_swiglu_down_native_vnni_decode_runtime(
        const float *const *d_gate_ptrs,
        const float *const *d_up_ptrs,
        const void *d_runtime_layer,
        const int *d_expert_ids,
        const float *d_weights,
        int8_t *d_swiglu_int8,
        float *d_swiglu_scales,
        float *d_output,
        int num_active,
        int d_model,
        int intermediate,
        int num_experts,
        uint8_t codebook_id,
        int device_idx,
        void *stream);

    bool cudaMoE_grouped_swiglu_down_native_vnni_decode_runtime_kpart(
        const float *const *d_gate_ptrs,
        const float *const *d_up_ptrs,
        const void *d_runtime_layer,
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
        const int *d_original_expert_ids,
        const int *d_active_expert_ids,
        const float *d_group_weights,
        int8_t *d_scratch_A_int8,
        float *d_scratch_scales,
        float *d_scratch_gate,
        float *d_scratch_up,
        float *d_gate_partials,
        float *d_up_partials,
        int8_t *d_scratch_swiglu_int8,
        float *d_scratch_swiglu_scales,
        float *d_down_partials,
        float *d_scratch_down_out,
        float *d_output,
        int num_experts,
        int d_model,
        int intermediate,
        int max_tokens_per_expert,
        int total_slots,
        int top_k,
        int active_expert_slots,
        int grouped_indices_are_route_slots,
        uint8_t gateup_codebook_id,
        uint8_t down_codebook_id,
        uint32_t gateup_codebook_mask,
        uint32_t down_codebook_mask,
        int gateup_k_partitions,
        int down_k_partitions,
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
        // Workspace rebinding is common during graph warmup/capture because
        // multiple MoE stages share this singleton kernel.  A same-workspace
        // rebind must be a no-op: runtime pointer arrays are graph-capture
        // metadata, and clearing them here would erase the tables populated
        // by the warmup pass before the capture pass can consume them.
        //
        // Use DeviceWorkspaceManager::id(), not the host pointer alone.  The
        // allocator can destroy and recreate a manager at the same host address
        // while growing graph workspace requirements; treating that ABA pointer
        // reuse as a no-op would preserve stale device sub-buffer pointers.
        const uint64_t next_workspace_id = workspace ? workspace->id() : 0;
        if (workspace_ == workspace && bound_workspace_id_ == next_workspace_id)
        {
            const bool needs_descriptor_rebind =
                workspace_ &&
                (std::any_of(grouped_down_desc_tables_.begin(), grouped_down_desc_tables_.end(),
                             [](const GroupedDownDescriptorTable &table)
                             {
                                 return table.valid && !table.host_descs.empty() && !table.device_descs;
                             }) ||
                 std::any_of(grouped_gateup_desc_tables_.begin(), grouped_gateup_desc_tables_.end(),
                             [](const GroupedGateUpDescriptorTable &table)
                             {
                                 return table.valid &&
                                        !table.host_gate_descs.empty() &&
                                        !table.host_up_descs.empty() &&
                                        (!table.device_gate_descs || !table.device_up_descs);
                             }));
            if (needs_descriptor_rebind)
                (void)rebindGroupedDescriptorTablesToWorkspace("bindWorkspace");
            return;
        }

        CUDAKernelBase::bindWorkspace(workspace);
        bound_workspace_id_ = next_workspace_id;
        clearWorkspaceScratchBindings();
        if (workspace_)
            (void)rebindGroupedDescriptorTablesToWorkspace("bindWorkspace");
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
        const DeviceId expected = deviceId();
        if (workspace_->device() != expected)
        {
            LOG_ERROR("[CUDAMoEKernel] " << context << " requires " << expected.to_string()
                                         << " workspace, got " << workspace_->device().to_string()
                                         << " (workspace id=" << workspace_->id() << ")");
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
        if (!requireCudaDevicePointer(buffer, device_ordinal_, name, context, nullptr))
            return false;
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
        d_group_expert_mask_ = nullptr;
        group_active_expert_slots_ = 0;
        group_slots_cap_ = 0;
        group_experts_cap_ = 0;
        group_expert_mask_cap_ = 0;
        group_expert_mask_hash_ = 0;
        d_prefill_A_int8_ = nullptr;
        d_prefill_A_scales_ = nullptr;
        d_prefill_swiglu_int8_ = nullptr;
        d_prefill_swiglu_scales_ = nullptr;
        d_prefill_gate_ = nullptr;
        d_prefill_up_ = nullptr;
        prefill_slots_cap_ = 0;
        prefill_d_model_cap_ = 0;
        prefill_intermediate_cap_ = 0;
        d_runtime_prefill_gate_descs_ = nullptr;
        d_runtime_prefill_up_descs_ = nullptr;
        d_runtime_prefill_down_descs_ = nullptr;
        runtime_prefill_desc_cap_ = 0;
        d_decode_hidden_int8_ = nullptr;
        d_decode_hidden_scales_ = nullptr;
        router_q8_hidden_source_ = nullptr;
        router_q8_hidden_valid_ = false;
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
        d_routing_decode_expert_ids_ = nullptr;
        grouped_decode_metadata_cap_ = 0;
        routing_decode_metadata_cap_ = 0;
        grouped_decode_cached_expert_ids_.clear();
        grouped_decode_cached_weights_.clear();
        gateup_pointer_slot_ready_.fill(false);
        down_pointer_slot_ready_.fill(false);
        for (auto &table : grouped_down_desc_tables_)
        {
            table.device_descs = nullptr;
            table.workspace_slot = 0;
        }
        for (auto &table : grouped_gateup_desc_tables_)
        {
            table.device_gate_descs = nullptr;
            table.device_up_descs = nullptr;
            table.workspace_slot = 0;
        }
        router_q8_gate_cache_.clear();
        next_grouped_down_desc_workspace_slot_ = 0;
        next_grouped_gateup_desc_workspace_slot_ = 0;
        next_router_q8_gate_workspace_slot_ = 0;
        scratch_workspace_bound_ = false;
    }

    bool CUDAMoEKernel::rebindGroupedDescriptorTablesToWorkspace(const char *context)
    {
        if (grouped_down_desc_tables_.empty() && grouped_gateup_desc_tables_.empty())
            return true;
        if (!setMoEDevice(device_ordinal_, context ? context : "rebindGroupedDescriptorTablesToWorkspace"))
            return false;
        cudaStream_t stream = static_cast<cudaStream_t>(getStream());
        if (!stream)
        {
            LOG_ERROR("[CUDAMoEKernel::rebindGroupedDescriptorTablesToWorkspace] explicit CUDA stream is required");
            return false;
        }

        next_grouped_down_desc_workspace_slot_ = 0;
        for (auto &table : grouped_down_desc_tables_)
        {
            if (!table.valid || table.host_descs.empty() || table.num_experts <= 0)
                continue;
            const std::size_t slot = next_grouped_down_desc_workspace_slot_;
            if (slot >= static_cast<std::size_t>(MoEWorkspaceBuffers::kGroupedDescriptorTableSlots))
            {
                LOG_ERROR("[CUDAMoEKernel::rebindGroupedDescriptorTablesToWorkspace] down descriptor slots exhausted");
                table.device_descs = nullptr;
                return false;
            }
            const size_t desc_bytes =
                static_cast<size_t>(table.num_experts) * sizeof(DeviceNativeVNNIMatrixDesc);
            void *base = nullptr;
            const size_t table_bytes =
                static_cast<size_t>(MoEWorkspaceBuffers::kGroupedDescriptorTableSlots) * desc_bytes;
            if (!bindWorkspaceBuffer(&base,
                                     MoEWorkspaceBuffers::CUDA_GROUPED_DOWN_DESC_TABLES,
                                     table_bytes,
                                     "CUDA grouped down descriptor table rebind"))
            {
                table.device_descs = nullptr;
                return false;
            }
            auto *device_descs =
                static_cast<DeviceNativeVNNIMatrixDesc *>(base) +
                slot * static_cast<std::size_t>(table.num_experts);
            const cudaError_t err = cudaMemcpyAsync(device_descs,
                                                    table.host_descs.data(),
                                                    desc_bytes,
                                                    cudaMemcpyHostToDevice,
                                                    stream);
            if (err != cudaSuccess)
            {
                LOG_ERROR("[CUDAMoEKernel::rebindGroupedDescriptorTablesToWorkspace] down descriptor upload failed: "
                          << cudaGetErrorString(err));
                table.device_descs = nullptr;
                return false;
            }
            table.device_descs = device_descs;
            table.workspace_slot = slot;
            next_grouped_down_desc_workspace_slot_ = slot + 1;
        }

        next_grouped_gateup_desc_workspace_slot_ = 0;
        for (auto &table : grouped_gateup_desc_tables_)
        {
            if (!table.valid || table.host_gate_descs.empty() ||
                table.host_up_descs.empty() || table.num_experts <= 0)
            {
                continue;
            }
            const std::size_t slot = next_grouped_gateup_desc_workspace_slot_;
            if (slot >= static_cast<std::size_t>(MoEWorkspaceBuffers::kGroupedDescriptorTableSlots))
            {
                LOG_ERROR("[CUDAMoEKernel::rebindGroupedDescriptorTablesToWorkspace] gate/up descriptor slots exhausted");
                table.device_gate_descs = nullptr;
                table.device_up_descs = nullptr;
                return false;
            }
            const size_t desc_bytes =
                static_cast<size_t>(table.num_experts) * sizeof(DeviceNativeVNNIMatrixDesc);
            const size_t table_bytes =
                static_cast<size_t>(MoEWorkspaceBuffers::kGroupedDescriptorTableSlots) * desc_bytes;
            void *gate_base = nullptr;
            void *up_base = nullptr;
            if (!bindWorkspaceBuffer(&gate_base,
                                     MoEWorkspaceBuffers::CUDA_GROUPED_GATE_DESC_TABLES,
                                     table_bytes,
                                     "CUDA grouped gate descriptor table rebind") ||
                !bindWorkspaceBuffer(&up_base,
                                     MoEWorkspaceBuffers::CUDA_GROUPED_UP_DESC_TABLES,
                                     table_bytes,
                                     "CUDA grouped up descriptor table rebind"))
            {
                table.device_gate_descs = nullptr;
                table.device_up_descs = nullptr;
                return false;
            }
            auto *device_gate_descs =
                static_cast<DeviceNativeVNNIMatrixDesc *>(gate_base) +
                slot * static_cast<std::size_t>(table.num_experts);
            auto *device_up_descs =
                static_cast<DeviceNativeVNNIMatrixDesc *>(up_base) +
                slot * static_cast<std::size_t>(table.num_experts);
            cudaError_t err = cudaMemcpyAsync(device_gate_descs,
                                              table.host_gate_descs.data(),
                                              desc_bytes,
                                              cudaMemcpyHostToDevice,
                                              stream);
            if (err == cudaSuccess)
                err = cudaMemcpyAsync(device_up_descs,
                                      table.host_up_descs.data(),
                                      desc_bytes,
                                      cudaMemcpyHostToDevice,
                                      stream);
            if (err != cudaSuccess)
            {
                LOG_ERROR("[CUDAMoEKernel::rebindGroupedDescriptorTablesToWorkspace] gate/up descriptor upload failed: "
                          << cudaGetErrorString(err));
                table.device_gate_descs = nullptr;
                table.device_up_descs = nullptr;
                return false;
            }
            table.device_gate_descs = device_gate_descs;
            table.device_up_descs = device_up_descs;
            table.workspace_slot = slot;
            next_grouped_gateup_desc_workspace_slot_ = slot + 1;
        }
        return true;
    }

    void CUDAMoEKernel::resetDynamicState()
    {
        /*
         * This hook is reached only on hard kernel-dynamic reset boundaries.
         * Replay-preserving request reset deliberately avoids
         * KernelFactory::resetAllDynamicState(), because captured CUDA graphs
         * may still reference workspace-backed pointer tables.  Once the hard
         * reset path is chosen, grouped prefill/decode scratch, mask upload
         * hashes, router caches, and descriptor-table handles must all be
         * treated as invalid so cached stages rebuild them before the next
         * eager warmup or capture.
         */
        clearWorkspaceScratchBindings();
        grouped_down_desc_tables_.clear();
        grouped_gateup_desc_tables_.clear();
        host_expert_counts_.clear();
        host_expert_offsets_.clear();
        host_grouped_indices_.clear();
        host_grouped_weights_.clear();
        prepared_num_experts_ = 0;
    }

    void CUDAMoEKernel::releaseDeviceBuffers() noexcept
    {
        cudaSetDevice(device_ordinal_);
        clearWorkspaceScratchBindings();
        staging_capacity_ = 0;
        route_logits_capacity_ = 0;
        route_topk_capacity_ = 0;
        group_slots_cap_ = 0;
        group_experts_cap_ = 0;
        group_active_expert_slots_ = 0;
        group_expert_mask_cap_ = 0;
        group_expert_mask_hash_ = 0;
        prefill_slots_cap_ = 0;
        prefill_d_model_cap_ = 0;
        prefill_intermediate_cap_ = 0;
        runtime_prefill_desc_cap_ = 0;
        decode_gateup_topk_cap_ = 0;
        decode_gateup_d_model_cap_ = 0;
        router_q8_hidden_source_ = nullptr;
        router_q8_hidden_valid_ = false;
        router_q8_gate_cache_.clear();
        next_router_q8_gate_workspace_slot_ = 0;
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
        next_grouped_down_desc_workspace_slot_ = 0;
        next_grouped_gateup_desc_workspace_slot_ = 0;
        gateup_pointer_slot_ready_.fill(false);
        down_pointer_slot_ready_.fill(false);

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
        // Capacity alone is not enough for graph-captured MoE routing.  This
        // kernel is a per-device singleton shared by many stages, so a previous
        // eager or stale binding can leave counters that look large enough while
        // the actual pointers are no longer the current graph workspace buffers.
        if (route_buffers_workspace_bound_ &&
            d_route_logits_ && d_route_indices_ && d_route_weights_ &&
            logits_count <= route_logits_capacity_ && topk_count <= route_topk_capacity_)
        {
            return true;
        }

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
        if (!need_realloc &&
            d_prefill_A_int8_ &&
            d_prefill_A_scales_ &&
            d_prefill_swiglu_int8_ &&
            d_prefill_swiglu_scales_ &&
            d_prefill_gate_ &&
            d_prefill_up_)
        {
            return true;
        }

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

    bool CUDAMoEKernel::ensureRuntimePrefillDescriptorCapacity(int num_experts)
    {
        if (num_experts <= 0 || num_experts > kDeviceMoEMaxExperts)
            return false;
        if (d_runtime_prefill_gate_descs_ &&
            d_runtime_prefill_up_descs_ &&
            d_runtime_prefill_down_descs_ &&
            runtime_prefill_desc_cap_ >= num_experts)
        {
            return true;
        }

        void *gate_descs = nullptr;
        void *up_descs = nullptr;
        void *down_descs = nullptr;
        const size_t bytes =
            static_cast<size_t>(num_experts) * sizeof(DeviceNativeVNNIMatrixDesc);
        const bool ok =
            bindWorkspaceBuffer(&gate_descs, MoEWorkspaceBuffers::CUDA_RUNTIME_PREFILL_GATE_DESC_TABLE,
                                bytes, "runtime prefill gate descriptors") &&
            bindWorkspaceBuffer(&up_descs, MoEWorkspaceBuffers::CUDA_RUNTIME_PREFILL_UP_DESC_TABLE,
                                bytes, "runtime prefill up descriptors") &&
            bindWorkspaceBuffer(&down_descs, MoEWorkspaceBuffers::CUDA_RUNTIME_PREFILL_DOWN_DESC_TABLE,
                                bytes, "runtime prefill down descriptors");
        if (!ok)
        {
            d_runtime_prefill_gate_descs_ = nullptr;
            d_runtime_prefill_up_descs_ = nullptr;
            d_runtime_prefill_down_descs_ = nullptr;
            runtime_prefill_desc_cap_ = 0;
            return false;
        }

        d_runtime_prefill_gate_descs_ =
            static_cast<DeviceNativeVNNIMatrixDesc *>(gate_descs);
        d_runtime_prefill_up_descs_ =
            static_cast<DeviceNativeVNNIMatrixDesc *>(up_descs);
        d_runtime_prefill_down_descs_ =
            static_cast<DeviceNativeVNNIMatrixDesc *>(down_descs);
        runtime_prefill_desc_cap_ = num_experts;
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
                                static_cast<size_t>(MoEWorkspaceBuffers::kMaxVerifierRows) *
                                    static_cast<size_t>(d_model) * sizeof(int8_t),
                                "decode hidden int8") &&
            bindWorkspaceBuffer(&decode_hidden_scales, MoEWorkspaceBuffers::DECODE_HIDDEN_SCALES,
                                static_cast<size_t>(MoEWorkspaceBuffers::kMaxVerifierRows) *
                                    static_cast<size_t>(blocks_per_row) * sizeof(float),
                                "decode hidden scales");
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

    const CUDAMoEKernel::RouterQ8GateCacheEntry *CUDAMoEKernel::getOrCreateQ8RouterGateCache(
        ITensor *gate_weights,
        const float *gate_device_ptr,
        int d_model,
        int num_experts)
    {
        if (!debugEnv().gemm.cuda_moe_router_q8)
            return nullptr;
        if (!gate_weights || !gate_device_ptr || d_model <= 0 || num_experts <= 0 || (d_model % 32) != 0)
        {
            LOG_ERROR("[CUDAMoEKernel::getOrCreateQ8RouterGateCache] invalid Q8 router gate request "
                      "(gate_weights=" << gate_weights
                      << " gate_device_ptr=" << static_cast<const void *>(gate_device_ptr)
                      << " d_model=" << d_model
                      << " num_experts=" << num_experts << ")");
            return nullptr;
        }

        const auto tensor_key = reinterpret_cast<std::uintptr_t>(gate_weights);
        const auto host_ptr_key = reinterpret_cast<std::uintptr_t>(gate_weights->raw_data());
        const auto device_ptr_key = reinterpret_cast<std::uintptr_t>(gate_device_ptr);
        for (const auto &entry : router_q8_gate_cache_)
        {
            if (entry.source_tensor == tensor_key &&
                entry.source_host_ptr == host_ptr_key &&
                entry.source_device_ptr == device_ptr_key &&
                entry.d_model == d_model &&
                entry.num_experts == num_experts &&
                entry.d_gate_weights_q8 && entry.d_gate_scales)
            {
                return &entry;
            }
        }

        void *stream = getStream();
        const bool capture_active =
            isGraphCaptureActive() ||
            (deviceContext() && deviceContext()->isDeviceGraphCaptureActive()) ||
            isCudaStreamCapturing(stream);
        if (capture_active)
        {
            LOG_ERROR("[CUDAMoEKernel::getOrCreateQ8RouterGateCache] Q8 router cache miss during graph capture");
            return nullptr;
        }
        if (!stream)
        {
            LOG_ERROR("[CUDAMoEKernel::getOrCreateQ8RouterGateCache] explicit CUDA stream is required");
            return nullptr;
        }

        const size_t d_model_sz = static_cast<size_t>(d_model);
        const size_t experts_sz = static_cast<size_t>(num_experts);
        if (d_model_sz > std::numeric_limits<size_t>::max() / experts_sz)
        {
            LOG_ERROR("[CUDAMoEKernel::getOrCreateQ8RouterGateCache] router gate size overflow");
            return nullptr;
        }
        const size_t element_count = d_model_sz * experts_sz;
        const int blocks_per_row = d_model / 32;
        const size_t scale_count = static_cast<size_t>(num_experts) * static_cast<size_t>(blocks_per_row);

        if (!setMoEDevice(device_ordinal_, "getOrCreateQ8RouterGateCache"))
            return nullptr;

        for (auto it = router_q8_gate_cache_.begin(); it != router_q8_gate_cache_.end();)
        {
            if (it->source_tensor == tensor_key &&
                it->d_model == d_model &&
                it->num_experts == num_experts &&
                (it->source_device_ptr != device_ptr_key ||
                 it->source_host_ptr != host_ptr_key))
            {
                it = router_q8_gate_cache_.erase(it);
            }
            else
            {
                ++it;
            }
        }

        int8_t *d_gate_weights_q8 = nullptr;
        float *d_gate_scales = nullptr;
        const std::size_t slot = next_router_q8_gate_workspace_slot_;
        if (slot >= static_cast<std::size_t>(MoEWorkspaceBuffers::kRouterGateCacheSlots))
        {
            LOG_ERROR("[CUDAMoEKernel::getOrCreateQ8RouterGateCache] Q8 router gate workspace slots exhausted: "
                      << slot << " capacity=" << MoEWorkspaceBuffers::kRouterGateCacheSlots);
            return nullptr;
        }

        void *gate_weights_base = nullptr;
        void *gate_scales_base = nullptr;
        const size_t weights_bytes =
            static_cast<size_t>(MoEWorkspaceBuffers::kRouterGateCacheSlots) *
            element_count * sizeof(int8_t);
        const size_t scales_bytes =
            static_cast<size_t>(MoEWorkspaceBuffers::kRouterGateCacheSlots) *
            scale_count * sizeof(float);
        if (!bindWorkspaceBuffer(&gate_weights_base,
                                 MoEWorkspaceBuffers::CUDA_ROUTER_Q8_GATE_WEIGHTS,
                                 weights_bytes,
                                 "CUDA Q8 router gate weights") ||
            !bindWorkspaceBuffer(&gate_scales_base,
                                 MoEWorkspaceBuffers::CUDA_ROUTER_Q8_GATE_SCALES,
                                 scales_bytes,
                                 "CUDA Q8 router gate scales"))
        {
            return nullptr;
        }
        d_gate_weights_q8 =
            static_cast<int8_t *>(gate_weights_base) + slot * element_count;
        d_gate_scales =
            static_cast<float *>(gate_scales_base) + slot * scale_count;

        if (!cudaMoE_quantize_router_gate_q8(gate_device_ptr, d_gate_weights_q8, d_gate_scales,
                                             d_model, num_experts,
                                             device_ordinal_, stream))
        {
            LOG_ERROR("[CUDAMoEKernel::getOrCreateQ8RouterGateCache] FP32->Q8 router gate conversion launch failed");
            return nullptr;
        }

        RouterQ8GateCacheEntry entry{};
        entry.source_tensor = tensor_key;
        entry.source_host_ptr = host_ptr_key;
        entry.source_device_ptr = device_ptr_key;
        entry.d_model = d_model;
        entry.num_experts = num_experts;
        entry.blocks_per_row = blocks_per_row;
        entry.element_count = element_count;
        entry.scale_count = scale_count;
        entry.d_gate_weights_q8 = d_gate_weights_q8;
        entry.d_gate_scales = d_gate_scales;
        entry.workspace_slot = slot;
        router_q8_gate_cache_.push_back(entry);
        next_router_q8_gate_workspace_slot_ = slot + 1;

        LOG_DEBUG("[CUDAMoEKernel] Cached Q8 router gate tensor="
                  << reinterpret_cast<const void *>(tensor_key)
                  << " host="
                  << reinterpret_cast<const void *>(host_ptr_key)
                  << " source_device=" << reinterpret_cast<const void *>(device_ptr_key)
                  << " shape=[" << num_experts << "," << d_model << "] payload_bytes="
                  << element_count << " scale_bytes=" << (scale_count * sizeof(float)));
        return &router_q8_gate_cache_.back();
    }

    bool CUDAMoEKernel::tryRouteDecodeLogitsQ8(
        ITensor *gate_weights,
        const float *d_hidden,
        const float *d_gate,
        int d_model,
        int num_experts,
        int top_k,
        const char *context)
    {
        router_q8_hidden_source_ = nullptr;
        router_q8_hidden_valid_ = false;
        if (!debugEnv().gemm.cuda_moe_router_q8)
        {
            LOG_ERROR("[CUDAMoEKernel::tryRouteDecodeLogitsQ8] Q8 router requested while disabled");
            return false;
        }
        if (!d_hidden || !d_gate || d_model <= 0 || num_experts <= 0 || (d_model % 32) != 0)
        {
            LOG_ERROR("[CUDAMoEKernel::tryRouteDecodeLogitsQ8] invalid Q8 router shape or pointer state");
            return false;
        }
        if (!ensureGroupedGateUpDecodeCapacity(top_k, d_model))
        {
            LOG_ERROR("[CUDAMoEKernel::tryRouteDecodeLogitsQ8] Q8 router hidden scratch unavailable");
            return false;
        }

        const auto *q8_gate = getOrCreateQ8RouterGateCache(gate_weights, d_gate, d_model, num_experts);
        if (!q8_gate)
        {
            LOG_ERROR("[CUDAMoEKernel::tryRouteDecodeLogitsQ8] Q8 router gate cache unavailable");
            return false;
        }

        void *stream = requireStream(context ? context : "CUDAMoEKernel::tryRouteDecodeLogitsQ8");
        const bool ok = cudaMoE_gate_logits_single_token_q8_weights(
            d_hidden,
            d_decode_hidden_int8_,
            d_decode_hidden_scales_,
            q8_gate->d_gate_weights_q8,
            q8_gate->d_gate_scales,
            d_route_logits_,
            d_model,
            num_experts,
            device_ordinal_,
            stream);
        if (!ok)
        {
            LOG_ERROR("[CUDAMoEKernel::tryRouteDecodeLogitsQ8] Q8 router logits kernel failed");
            return false;
        }

        /*
         * During stream capture the logits/hidden-quant kernels are only
         * recorded; they have not produced bytes in d_decode_hidden_int8_ yet.
         * Leave the reuse marker invalid so a later host-side capture phase
         * cannot consume stale scratch as if the recorded kernel had run.
         */
        if (!isCudaMoEDecodeCaptureActive(stream))
        {
            router_q8_hidden_source_ = d_hidden;
            router_q8_hidden_valid_ = true;
        }
        PerfStatsCollector::addCounter(
            "kernel", "cuda_moe_router_q8_decode_calls", 1.0, {}, {},
            {{"num_experts", std::to_string(num_experts)},
             {"d_model", std::to_string(d_model)}});
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

        // Decode uses one logical output row, while grouped verifier prefill
        // needs one partial row per compact verifier route/token owner.  The
        // caller still passes slots=1 for ordinary decode, so this remains
        // backward compatible with the existing single-row scratch contract.
        if (d_grouped_down_partials_ &&
            grouped_down_kpart_partitions_cap_ >= k_partitions &&
            grouped_down_kpart_d_model_cap_ >= d_model &&
            grouped_down_kpart_slots_cap_ >= slots)
            return true;

        const size_t partial_count =
            static_cast<size_t>(slots) *
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
        grouped_down_kpart_slots_cap_ = slots;
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

    bool CUDAMoEKernel::ensureGroupedDecodeMetadataCapacity(int num_active, bool include_weights)
    {
        if (num_active <= 0 ||
            num_active > static_cast<int>(kRuntimePointerArrayMaxTopK))
        {
            return false;
        }

        const bool capacity_ok =
            d_grouped_decode_expert_ids_ &&
            grouped_decode_metadata_cap_ >= num_active &&
            (!include_weights || d_grouped_decode_weights_);
        if (capacity_ok)
            return true;

        if (!setMoEDevice(device_ordinal_, "ensureGroupedDecodeMetadataCapacity"))
            return false;

        void *decode_expert_ids = nullptr;
        void *decode_weights = d_grouped_decode_weights_;
        bool ok = bindWorkspaceBuffer(
            &decode_expert_ids,
            MoEWorkspaceBuffers::DECODE_EXPERT_IDS,
            static_cast<size_t>(num_active) * sizeof(int),
            "decode expert ids");
        if (ok && include_weights)
        {
            ok = bindWorkspaceBuffer(
                &decode_weights,
                MoEWorkspaceBuffers::DECODE_WEIGHTS,
                static_cast<size_t>(num_active) * sizeof(float),
                "decode weights");
        }
        if (!ok)
        {
            d_grouped_decode_expert_ids_ = nullptr;
            if (include_weights)
                d_grouped_decode_weights_ = nullptr;
            grouped_decode_metadata_cap_ = 0;
            return false;
        }

        d_grouped_decode_expert_ids_ = static_cast<int *>(decode_expert_ids);
        if (include_weights)
            d_grouped_decode_weights_ = static_cast<float *>(decode_weights);
        grouped_decode_metadata_cap_ = num_active;

        /*
         * Device-generated metadata invalidates the host-upload cache used by
         * groupedExpert*FromTable().  Clearing it prevents a later host-table
         * call from assuming the workspace still contains cached host ids.
         */
        grouped_decode_cached_expert_ids_.clear();
        grouped_decode_cached_weights_.clear();
        return true;
    }

    bool CUDAMoEKernel::ensureRoutingDecodeMetadataCapacity(int num_active)
    {
        if (num_active <= 0 ||
            num_active > static_cast<int>(kRuntimePointerArrayMaxTopK))
        {
            return false;
        }
        if (d_routing_decode_expert_ids_ &&
            routing_decode_metadata_cap_ >= num_active)
        {
            return true;
        }

        void *routing_expert_ids = nullptr;
        if (!bindWorkspaceBuffer(
                &routing_expert_ids,
                MoEWorkspaceBuffers::CUDA_ROUTING_DECODE_EXPERT_IDS,
                static_cast<size_t>(num_active) * sizeof(int),
                "CUDA routing decode expert ids"))
        {
            d_routing_decode_expert_ids_ = nullptr;
            routing_decode_metadata_cap_ = 0;
            return false;
        }

        d_routing_decode_expert_ids_ = static_cast<int *>(routing_expert_ids);
        routing_decode_metadata_cap_ = num_active;
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

    bool CUDAMoEKernel::runtimePointerWorkspaceSlot(
        int table_id,
        RuntimePointerArrayScope scope,
        std::size_t *workspace_slot,
        const char *context) const
    {
        if (!workspace_slot || table_id < 0)
            return false;

        const std::size_t table_slot = static_cast<std::size_t>(table_id);
        if (table_slot >= kRuntimePointerArrayTableSlots)
        {
            LOG_ERROR("[CUDAMoEKernel] " << context
                                         << " pointer workspace table slot exceeded: table_id="
                                         << table_id << " table_slots="
                                         << kRuntimePointerArrayTableSlots);
            return false;
        }

        const std::size_t scope_slot = static_cast<std::size_t>(scope);
        const std::size_t slot =
            scope_slot * kRuntimePointerArrayTableSlots + table_slot;
        if (slot >= kRuntimePointerArrayWorkspaceEntries)
        {
            LOG_ERROR("[CUDAMoEKernel] " << context
                                         << " pointer workspace slot exceeded: scope="
                                         << scope_slot << " table_id=" << table_id
                                         << " capacity="
                                         << kRuntimePointerArrayWorkspaceEntries);
            return false;
        }

        *workspace_slot = slot;
        return true;
    }

    bool CUDAMoEKernel::ensureRuntimeGateUpPointerArrays(
        int table_id,
        RuntimePointerArrayScope scope,
        int top_k,
        const std::array<float *, CUDAMoEKernel::kRuntimePointerArrayMaxTopK> &gate_ptrs,
        const std::array<float *, CUDAMoEKernel::kRuntimePointerArrayMaxTopK> &up_ptrs,
        float ***d_gate_ptrs,
        float ***d_up_ptrs)
    {
        if (!d_gate_ptrs || !d_up_ptrs || table_id < 0 || top_k <= 0 ||
            top_k > static_cast<int>(kRuntimePointerArrayMaxTopK))
            return false;

        std::size_t workspace_slot = 0;
        if (!runtimePointerWorkspaceSlot(
                table_id, scope, &workspace_slot,
                "grouped gate/up"))
            return false;
        if (!setMoEDevice(device_ordinal_, "ensureRuntimeGateUpPointerArrays"))
            return false;

        void *gate_ptr_workspace = nullptr;
        void *up_ptr_workspace = nullptr;
        const size_t workspace_bytes =
            kRuntimePointerArrayWorkspaceEntries *
            kRuntimePointerArrayMaxTopK *
            sizeof(float *);
        if (!bindWorkspaceBuffer(&gate_ptr_workspace,
                                 MoEWorkspaceBuffers::CUDA_DECODE_GATEUP_GATE_PTRS,
                                 workspace_bytes,
                                 "CUDA grouped gate/up gate pointer arrays") ||
            !bindWorkspaceBuffer(&up_ptr_workspace,
                                 MoEWorkspaceBuffers::CUDA_DECODE_GATEUP_UP_PTRS,
                                 workspace_bytes,
                                 "CUDA grouped gate/up up pointer arrays"))
        {
            return false;
        }

        float **slot_gate_ptrs =
            static_cast<float **>(gate_ptr_workspace) +
            workspace_slot * kRuntimePointerArrayMaxTopK;
        float **slot_up_ptrs =
            static_cast<float **>(up_ptr_workspace) +
            workspace_slot * kRuntimePointerArrayMaxTopK;

        cudaStream_t stream = static_cast<cudaStream_t>(
            requireStream("CUDAMoEKernel::ensureRuntimeGateUpPointerArrays"));
        if (isCudaMoEDecodeCaptureActive(stream))
        {
            if (!gateup_pointer_slot_ready_[workspace_slot])
            {
                LOG_ERROR("[CUDAMoEKernel] grouped gate/up pointer workspace slot "
                          << workspace_slot << " was not staged before graph capture");
                return false;
            }
            *d_gate_ptrs = slot_gate_ptrs;
            *d_up_ptrs = slot_up_ptrs;
            return true;
        }

        cudaError_t err = cudaMemcpyAsync(slot_gate_ptrs, gate_ptrs.data(),
                                          static_cast<size_t>(top_k) * sizeof(float *),
                                          cudaMemcpyHostToDevice, stream);
        if (err == cudaSuccess)
            err = cudaMemcpyAsync(slot_up_ptrs, up_ptrs.data(),
                                  static_cast<size_t>(top_k) * sizeof(float *),
                                  cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel] grouped gate/up pointer staging failed: "
                      << cudaGetErrorString(err));
            return false;
        }

        gateup_pointer_slot_ready_[workspace_slot] = true;
        *d_gate_ptrs = slot_gate_ptrs;
        *d_up_ptrs = slot_up_ptrs;
        return true;
    }

    bool CUDAMoEKernel::ensureRuntimeDownPointerArrays(
        int table_id,
        RuntimePointerArrayScope scope,
        int top_k,
        const std::array<const float *, CUDAMoEKernel::kRuntimePointerArrayMaxTopK> &gate_ptrs,
        const std::array<const float *, CUDAMoEKernel::kRuntimePointerArrayMaxTopK> &up_ptrs,
        const float ***d_gate_ptrs,
        const float ***d_up_ptrs)
    {
        if (!d_gate_ptrs || !d_up_ptrs || table_id < 0 || top_k <= 0 ||
            top_k > static_cast<int>(kRuntimePointerArrayMaxTopK))
            return false;

        std::size_t workspace_slot = 0;
        if (!runtimePointerWorkspaceSlot(
                table_id, scope, &workspace_slot,
                "grouped down"))
            return false;
        if (!setMoEDevice(device_ordinal_, "ensureRuntimeDownPointerArrays"))
            return false;

        void *gate_ptr_workspace = nullptr;
        void *up_ptr_workspace = nullptr;
        const size_t workspace_bytes =
            kRuntimePointerArrayWorkspaceEntries *
            kRuntimePointerArrayMaxTopK *
            sizeof(const float *);
        if (!bindWorkspaceBuffer(&gate_ptr_workspace,
                                 MoEWorkspaceBuffers::CUDA_DECODE_DOWN_GATE_PTRS,
                                 workspace_bytes,
                                 "CUDA grouped down gate pointer arrays") ||
            !bindWorkspaceBuffer(&up_ptr_workspace,
                                 MoEWorkspaceBuffers::CUDA_DECODE_DOWN_UP_PTRS,
                                 workspace_bytes,
                                 "CUDA grouped down up pointer arrays"))
        {
            return false;
        }

        const float **slot_gate_ptrs =
            static_cast<const float **>(gate_ptr_workspace) +
            workspace_slot * kRuntimePointerArrayMaxTopK;
        const float **slot_up_ptrs =
            static_cast<const float **>(up_ptr_workspace) +
            workspace_slot * kRuntimePointerArrayMaxTopK;

        cudaStream_t stream = static_cast<cudaStream_t>(
            requireStream("CUDAMoEKernel::ensureRuntimeDownPointerArrays"));
        if (isCudaMoEDecodeCaptureActive(stream))
        {
            if (!down_pointer_slot_ready_[workspace_slot])
            {
                LOG_ERROR("[CUDAMoEKernel] grouped down pointer workspace slot "
                          << workspace_slot << " was not staged before graph capture");
                return false;
            }
            *d_gate_ptrs = slot_gate_ptrs;
            *d_up_ptrs = slot_up_ptrs;
            return true;
        }

        cudaError_t err = cudaMemcpyAsync(slot_gate_ptrs, gate_ptrs.data(),
                                          static_cast<size_t>(top_k) * sizeof(const float *),
                                          cudaMemcpyHostToDevice, stream);
        if (err == cudaSuccess)
            err = cudaMemcpyAsync(slot_up_ptrs, up_ptrs.data(),
                                  static_cast<size_t>(top_k) * sizeof(const float *),
                                  cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel] grouped down pointer staging failed: "
                      << cudaGetErrorString(err));
            return false;
        }

        down_pointer_slot_ready_[workspace_slot] = true;
        *d_gate_ptrs = slot_gate_ptrs;
        *d_up_ptrs = slot_up_ptrs;
        return true;
    }

    bool CUDAMoEKernel::routeCore(const float *hidden, const void *gate_weights, TensorType gate_type,
                                  int seq_len, int d_model, int num_experts, int top_k,
                                  bool normalize_weights, DeviceRouteBuffers &buffers,
                                  const int *device_effective_seq_len)
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
        if (device_effective_seq_len &&
            !requireCudaDevicePointer(device_effective_seq_len, device_ordinal_,
                                      "effective prefill sequence length",
                                      "routeCore", stream))
        {
            return false;
        }

        const bool route_ok = gate_is_fp32
                                  ? cudaMoE_route_logits(hidden, static_cast<const float *>(gate_weights), d_route_logits_,
                                                         seq_len, d_model, num_experts,
                                                         device_ordinal_, stream)
                                  : cudaMoE_route_logits_bf16(hidden, gate_weights, d_route_logits_,
                                                              seq_len, d_model, num_experts,
                                                              device_ordinal_, stream);
        if (!route_ok)
            return false;
        if (gate_is_fp32 && seq_len > 1)
        {
            PerfStatsCollector::addCounter(
                "kernel", "cuda_moe_router_tiled_prefill_calls", 1.0, {}, {},
                {{"seq_len", std::to_string(seq_len)},
                 {"d_model", std::to_string(d_model)},
                 {"num_experts", std::to_string(num_experts)}});
        }
        if (!cudaMoE_softmax_topk(d_route_logits_, d_route_indices_, d_route_weights_,
                                  seq_len, num_experts, top_k, normalize_weights,
                                  device_ordinal_, getStream(),
                                  device_effective_seq_len))
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

    bool CUDAMoEKernel::routeWithTensorsImpl(ITensor *hidden, ITensor *gate_weights,
                                             int seq_len, int d_model, int num_experts, int top_k,
                                             bool normalize_weights,
                                             ITensor *output_indices, ITensor *output_weights,
                                             MoERoutingResult &host_result,
                                             const int *device_effective_seq_len,
                                             const char *context)
    {
        void *stream = requireStream(context);
        const DeviceId device = deviceId();
        if (!ensureTensorOnDevice(hidden, device, stream, "hidden") ||
            !ensureTensorOnDevice(gate_weights, device, stream, "gate_weights") ||
            !ensureOutputOnDevice(output_indices, device, stream, "output_indices") ||
            !ensureOutputOnDevice(output_weights, device, stream, "output_weights"))
            return false;

        if (seq_len <= 0 || d_model <= 0 || num_experts <= 0 || top_k <= 0 || top_k > num_experts)
        {
            LOG_ERROR("[" << context << "] invalid routing shape seq_len="
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
                       seq_len, d_model, num_experts, top_k, normalize_weights, buffers,
                       device_effective_seq_len))
            return false;

        if (!cudaMoE_int_to_float(buffers.d_indices, d_idx, static_cast<int>(buffers.topk_count), device_ordinal_, stream))
            return false;
        cudaError_t err = cudaMemcpyAsync(d_wt, buffers.d_weights,
                                          buffers.topk_count * sizeof(float),
                                          cudaMemcpyDeviceToDevice,
                                          static_cast<cudaStream_t>(stream));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[" << context << "] D2D weights failed: " << cudaGetErrorString(err));
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
                LOG_ERROR("[" << context << "] stream sync failed: " << cudaGetErrorString(err));
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

    bool CUDAMoEKernel::routeWithTensors(ITensor *hidden, ITensor *gate_weights,
                                         int seq_len, int d_model, int num_experts, int top_k,
                                         bool normalize_weights,
                                         ITensor *output_indices, ITensor *output_weights,
                                         MoERoutingResult &host_result)
    {
        return routeWithTensorsImpl(hidden, gate_weights,
                                    seq_len, d_model, num_experts, top_k,
                                    normalize_weights,
                                    output_indices, output_weights,
                                    host_result,
                                    nullptr,
                                    "CUDAMoEKernel::routeWithTensors");
    }

    bool CUDAMoEKernel::routeWithTensorsEffectiveSeqLen(
        ITensor *hidden, ITensor *gate_weights,
        int seq_len, int d_model, int num_experts, int top_k,
        bool normalize_weights,
        ITensor *output_indices, ITensor *output_weights,
        MoERoutingResult &host_result,
        const int *device_effective_seq_len)
    {
        if (!device_effective_seq_len)
        {
            LOG_ERROR("[CUDAMoEKernel::routeWithTensorsEffectiveSeqLen] missing device effective length scalar");
            return false;
        }
        return routeWithTensorsImpl(hidden, gate_weights,
                                    seq_len, d_model, num_experts, top_k,
                                    normalize_weights,
                                    output_indices, output_weights,
                                    host_result,
                                    device_effective_seq_len,
                                    "CUDAMoEKernel::routeWithTensorsEffectiveSeqLen");
    }

    bool CUDAMoEKernel::routeVerifierRowsDecodeEquivalent(
        ITensor *hidden, ITensor *gate_weights,
        int seq_len, int d_model, int num_experts, int top_k,
        bool normalize_weights,
        ITensor *output_indices, ITensor *output_weights)
    {
        constexpr const char *kContext = "CUDAMoEKernel::routeVerifierRowsDecodeEquivalent";
        void *stream = requireStream(kContext);
        const DeviceId device = deviceId();
        if (!ensureTensorOnDevice(hidden, device, stream, "hidden") ||
            !ensureTensorOnDevice(gate_weights, device, stream, "gate_weights") ||
            !ensureOutputOnDevice(output_indices, device, stream, "output_indices") ||
            !ensureOutputOnDevice(output_weights, device, stream, "output_weights"))
        {
            return false;
        }

        if (seq_len < 1 || seq_len > 4 || d_model <= 0 ||
            num_experts <= 0 || top_k <= 0 || top_k > num_experts)
        {
            LOG_ERROR("[" << kContext << "] invalid verifier routing shape seq_len="
                          << seq_len << " d_model=" << d_model
                          << " num_experts=" << num_experts
                          << " top_k=" << top_k);
            return false;
        }

        auto *hidden_base = asTensorBase(hidden, "verifier route hidden");
        auto *gate_base = asTensorBase(gate_weights, "verifier route gate_weights");
        auto *indices_base = asTensorBase(output_indices, "verifier route output_indices");
        auto *weights_base = asTensorBase(output_weights, "verifier route output_weights");
        if (!hidden_base || !gate_base || !indices_base || !weights_base)
            return false;

        if (!requireTensorType(hidden_base, TensorType::FP32, "hidden", kContext) ||
            !requireTensorTypeOneOf(gate_base, TensorType::FP32, TensorType::BF16, "gate_weights", kContext) ||
            !requireTensorType(indices_base, TensorType::FP32, "output_indices", kContext) ||
            !requireTensorType(weights_base, TensorType::FP32, "output_weights", kContext))
        {
            return false;
        }

        const size_t required_hidden =
            static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
        const size_t required_gate =
            static_cast<size_t>(num_experts) * static_cast<size_t>(d_model);
        const size_t required_topk =
            static_cast<size_t>(seq_len) * static_cast<size_t>(top_k);
        if (!requireMatrixCapacity(hidden, seq_len, d_model, "hidden", kContext) ||
            !requireMatrixCapacity(gate_weights, num_experts, d_model, "gate_weights", kContext) ||
            !requireTensorElements(hidden, required_hidden, "hidden", kContext) ||
            !requireTensorElements(gate_weights, required_gate, "gate_weights", kContext) ||
            !requireTensorElements(output_indices, required_topk, "output_indices", kContext) ||
            !requireTensorElements(output_weights, required_topk, "output_weights", kContext))
        {
            return false;
        }

        const float *d_hidden = static_cast<const float *>(hidden->gpu_data_ptr());
        const void *d_gate = gate_weights->gpu_data_ptr();
        float *d_idx = static_cast<float *>(output_indices->gpu_data_ptr());
        float *d_wt = static_cast<float *>(output_weights->gpu_data_ptr());
        if (!d_hidden || !d_gate || !d_idx || !d_wt)
        {
            LOG_ERROR("[" << kContext << "] null device pointer hidden="
                          << static_cast<const void *>(d_hidden)
                          << " gate=" << d_gate
                          << " indices=" << static_cast<void *>(d_idx)
                          << " weights=" << static_cast<void *>(d_wt));
            return false;
        }

        const bool gate_is_fp32 = (gate_base->native_type() == TensorType::FP32);
        const bool q8_router_requested = gate_is_fp32 && debugEnv().gemm.cuda_moe_router_q8;
        if (q8_router_requested && (d_model % 32) != 0)
        {
            LOG_ERROR("[" << kContext << "] Q8 verifier router requires d_model to be a multiple of 32, got "
                          << d_model);
            return false;
        }

        const size_t logits_count =
            static_cast<size_t>(seq_len) * static_cast<size_t>(num_experts);
        const size_t topk_count =
            static_cast<size_t>(seq_len) * static_cast<size_t>(top_k);
        if (!ensureRouteBufferCapacity(logits_count, topk_count))
        {
            LOG_ERROR("[" << kContext << "] route scratch allocation failed");
            return false;
        }

        /*
         * Route all verifier rows as one grouped transaction while preserving
         * serial decode math inside each row.  The Q8, FP32, and BF16 kernels
         * all use expert-owned grouped blocks that share gate loads across
         * verifier rows while preserving each row's serial K/reduction order.
         */
        if (q8_router_requested)
        {
            if (!ensureGroupedGateUpDecodeCapacity(top_k, d_model))
            {
                LOG_ERROR("[" << kContext << "] Q8 router hidden scratch unavailable");
                return false;
            }
            const auto *q8_gate = getOrCreateQ8RouterGateCache(
                gate_weights, static_cast<const float *>(d_gate), d_model, num_experts);
            if (!q8_gate)
            {
                LOG_ERROR("[" << kContext << "] Q8 router gate cache unavailable");
                return false;
            }
            if (!cudaMoE_gate_logits_q8_weights_decode_equivalent_rows(
                    d_hidden,
                    d_decode_hidden_int8_,
                    d_decode_hidden_scales_,
                    q8_gate->d_gate_weights_q8,
                    q8_gate->d_gate_scales,
                    d_route_logits_,
                    seq_len,
                    d_model,
                    num_experts,
                    device_ordinal_,
                    stream))
            {
                LOG_ERROR("[" << kContext << "] grouped Q8 decode-equivalent logits failed");
                return false;
            }

            /*
             * Single-row verifier buckets are ordinary decode and may reuse the
             * router Q8 hidden scratch in the following gate/up decode.  For
             * M>1 the scratch is laid out as [M,d_model], so a one-row gate/up
             * reuse marker would be ambiguous; keep it invalid until the expert
             * grouped path has an explicit row-slice reuse contract.
             */
            if (seq_len == 1 && !isCudaMoEDecodeCaptureActive(stream))
            {
                router_q8_hidden_source_ = d_hidden;
                router_q8_hidden_valid_ = true;
            }
            else
            {
                router_q8_hidden_source_ = nullptr;
                router_q8_hidden_valid_ = false;
            }
        }
        else if (gate_is_fp32)
        {
            if (!cudaMoE_route_logits_decode_equivalent_rows(
                    d_hidden,
                    static_cast<const float *>(d_gate),
                    d_route_logits_,
                    seq_len,
                    d_model,
                    num_experts,
                    device_ordinal_,
                    stream))
            {
                LOG_ERROR("[" << kContext << "] grouped FP32 decode-equivalent logits failed");
                return false;
            }
        }
        else if (!cudaMoE_route_logits_bf16_decode_equivalent_rows(
                     d_hidden,
                     d_gate,
                     d_route_logits_,
                     seq_len,
                     d_model,
                     num_experts,
                     device_ordinal_,
                     stream))
        {
            LOG_ERROR("[" << kContext << "] grouped BF16 decode-equivalent logits failed");
            return false;
        }

        if (!cudaMoE_softmax_topk(
                d_route_logits_,
                d_route_indices_,
                d_route_weights_,
                seq_len,
                num_experts,
                top_k,
                normalize_weights,
                device_ordinal_,
                stream,
                nullptr))
        {
            LOG_ERROR("[" << kContext << "] grouped decode-equivalent softmax/top-k failed");
            return false;
        }

        if (!cudaMoE_int_to_float(d_route_indices_,
                                  d_idx,
                                  static_cast<int>(topk_count),
                                  device_ordinal_,
                                  stream))
        {
            LOG_ERROR("[" << kContext << "] grouped int-to-float index conversion failed");
            return false;
        }

        const cudaError_t copy_status = cudaMemcpyAsync(
            d_wt,
            d_route_weights_,
            topk_count * sizeof(float),
            cudaMemcpyDeviceToDevice,
            static_cast<cudaStream_t>(stream));
        if (copy_status != cudaSuccess)
        {
            LOG_ERROR("[" << kContext << "] grouped D2D weight copy failed: "
                          << cudaGetErrorString(copy_status));
            return false;
        }

        markDeviceWritten(output_indices, device, stream);
        markDeviceWritten(output_weights, device, stream);
        PerfStatsCollector::addCounter(
            "kernel",
            "cuda_moe_router_decode_equivalent_small_m_calls",
            1.0,
            {},
            {},
            {{"seq_len", std::to_string(seq_len)},
             {"d_model", std::to_string(d_model)},
             {"num_experts", std::to_string(num_experts)},
             {"top_k", std::to_string(top_k)},
             {"route", q8_router_requested ? "grouped_decode_equivalent_q8"
                                            : "grouped_decode_equivalent"}});
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

        bool logits_ready = false;
        const bool q8_router_requested = gate_is_fp32 && debugEnv().gemm.cuda_moe_router_q8;
        if (q8_router_requested && (d_model % 32) != 0)
        {
            LOG_ERROR("[CUDAMoEKernel::decodeRouteSelect] Q8 router requires d_model to be a multiple of 32, got "
                      << d_model);
            return false;
        }
        if (q8_router_requested)
        {
            if (!tryRouteDecodeLogitsQ8(
                gate_weights, d_hidden, static_cast<const float *>(d_gate),
                d_model, num_experts, top_k, "CUDAMoEKernel::decodeRouteSelect"))
            {
                return false;
            }
            logits_ready = true;
        }
        if (!logits_ready)
        {
            router_q8_hidden_source_ = nullptr;
            router_q8_hidden_valid_ = false;
            const bool route_ok = gate_is_fp32
                                      ? cudaMoE_route_logits(d_hidden, static_cast<const float *>(d_gate), d_route_logits_,
                                                             /*seq_len=*/1, d_model, num_experts,
                                                             device_ordinal_, stream)
                                      : cudaMoE_route_logits_bf16(d_hidden, d_gate, d_route_logits_,
                                                                  /*seq_len=*/1, d_model, num_experts,
                                                                  device_ordinal_, stream);
            if (!route_ok)
                return false;
        }
        if (!cudaMoE_softmax_topk_decode_runtime(d_route_logits_,
                                                 runtime_layer,
                                                 legacy_indices, legacy_weights,
                                                 num_experts, top_k, normalize_weights,
                                                 write_legacy_outputs, update_runtime_histogram,
                                                 nullptr,
                                                 nullptr,
                                                 0u,
                                                 nullptr,
                                                 nullptr,
                                                 0u,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 -1,
                                                 1u,
                                                 device_ordinal_, stream))
            return false;

        if (write_legacy_outputs)
        {
            markDeviceWritten(output_indices, device, stream);
            markDeviceWritten(output_weights, device, stream);
        }
        return true;
    }

    bool CUDAMoEKernel::decodeRouteSelectWithReadyRebalanceApply(
        DeviceMoELayerRuntime *runtime_layers,
        DeviceMoELayerRuntime *runtime_layer,
        ITensor *hidden, ITensor *gate_weights,
        int d_model, int num_experts, int top_k,
        bool normalize_weights,
        ITensor *output_indices, ITensor *output_weights,
        bool write_legacy_outputs,
        bool update_runtime_histogram,
        const DeviceMoERebalancePlanEntry *rebalance_plan_entries,
        uint32_t rebalance_plan_capacity,
        DeviceMoERebalanceCommandBufferHeader *rebalance_command_header,
        const DeviceMoEExpertDirectoryEntry *rebalance_local_transfer_slots,
        uint32_t rebalance_local_transfer_slot_count,
        const DeviceMoERebalanceConfig &rebalance_config,
        DeviceMoERebalanceApplyStatus *rebalance_apply_status,
        DeviceMoERebalanceGraphControllerState *rebalance_controller_state,
        int rebalance_target_layer,
        uint32_t rebalance_command_buffer_count)
    {
        void *stream = requireStream("CUDAMoEKernel::decodeRouteSelectWithReadyRebalanceApply");
        const DeviceId device = deviceId();
        if (!runtime_layers || !runtime_layer || !hidden || !gate_weights ||
            !rebalance_plan_entries || rebalance_plan_capacity == 0 ||
            !rebalance_command_header || !rebalance_apply_status ||
            !rebalance_controller_state ||
            !validateDeviceMoERebalanceConfig(rebalance_config))
        {
            LOG_ERROR("[CUDAMoEKernel::decodeRouteSelectWithReadyRebalanceApply] invalid runtime or rebalance binding");
            return false;
        }
        if (!ensureTensorOnDevice(hidden, device, stream, "hidden") ||
            !ensureTensorOnDevice(gate_weights, device, stream, "gate_weights"))
            return false;

        auto *hidden_base = asTensorBase(hidden, "decodeRouteSelectWithReadyRebalanceApply hidden");
        auto *gate_base = asTensorBase(gate_weights, "decodeRouteSelectWithReadyRebalanceApply gate_weights");
        if (!hidden_base || !gate_base)
            return false;
        if (hidden_base->native_type() != TensorType::FP32)
        {
            LOG_ERROR("[CUDAMoEKernel::decodeRouteSelectWithReadyRebalanceApply] hidden must be FP32, got "
                      << tensorTypeName(hidden_base->native_type()));
            return false;
        }
        const TensorType gate_type = gate_base->native_type();
        const bool gate_is_fp32 = (gate_type == TensorType::FP32);
        const bool gate_is_bf16 = (gate_type == TensorType::BF16);
        if (!gate_is_fp32 && !gate_is_bf16)
        {
            LOG_ERROR("[CUDAMoEKernel::decodeRouteSelectWithReadyRebalanceApply] CUDA runtime router supports FP32 or BF16 gate weights, got "
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
        if (!requireAlignedPointer(d_hidden, 16, "hidden", "decodeRouteSelectWithReadyRebalanceApply") ||
            !requireAlignedPointer(d_route_logits_, 16, "route logits", "decodeRouteSelectWithReadyRebalanceApply"))
            return false;
        if (gate_is_fp32 && !requireAlignedPointer(d_gate, 16, "FP32 gate", "decodeRouteSelectWithReadyRebalanceApply"))
            return false;

        bool logits_ready = false;
        const bool q8_router_requested = gate_is_fp32 && debugEnv().gemm.cuda_moe_router_q8;
        if (q8_router_requested && (d_model % 32) != 0)
        {
            LOG_ERROR("[CUDAMoEKernel::decodeRouteSelectWithReadyRebalanceApply] Q8 router requires d_model to be a multiple of 32, got "
                      << d_model);
            return false;
        }
        if (q8_router_requested)
        {
            if (!tryRouteDecodeLogitsQ8(
                gate_weights, d_hidden, static_cast<const float *>(d_gate),
                d_model, num_experts, top_k,
                "CUDAMoEKernel::decodeRouteSelectWithReadyRebalanceApply"))
            {
                return false;
            }
            logits_ready = true;
        }
        if (!logits_ready)
        {
            router_q8_hidden_source_ = nullptr;
            router_q8_hidden_valid_ = false;
            const bool route_ok = gate_is_fp32
                                      ? cudaMoE_route_logits(d_hidden, static_cast<const float *>(d_gate), d_route_logits_,
                                                             /*seq_len=*/1, d_model, num_experts,
                                                             device_ordinal_, stream)
                                      : cudaMoE_route_logits_bf16(d_hidden, d_gate, d_route_logits_,
                                                                  /*seq_len=*/1, d_model, num_experts,
                                                                  device_ordinal_, stream);
            if (!route_ok)
                return false;
        }
        if (!cudaMoE_softmax_topk_decode_runtime(
                d_route_logits_,
                runtime_layer,
                legacy_indices, legacy_weights,
                num_experts, top_k, normalize_weights,
                write_legacy_outputs, update_runtime_histogram,
                runtime_layers,
                rebalance_plan_entries,
                rebalance_plan_capacity,
                rebalance_command_header,
                rebalance_local_transfer_slots,
                rebalance_local_transfer_slot_count,
                &rebalance_config,
                rebalance_apply_status,
                rebalance_controller_state,
                rebalance_target_layer,
                rebalance_command_buffer_count,
                device_ordinal_, stream))
            return false;

        if (write_legacy_outputs)
        {
            markDeviceWritten(output_indices, device, stream);
            markDeviceWritten(output_weights, device, stream);
        }
        return true;
    }

    bool CUDAMoEKernel::runDeviceRebalanceController(
        DeviceMoELayerRuntime *runtime_layers,
        const uint64_t *gathered_histograms,
        DeviceMoERebalanceStatus *status,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalancePlanEntry *plan_entries,
        uint32_t *plan_count,
        uint32_t plan_capacity,
        uint32_t payload_slot_capacity,
        DeviceMoERebalanceCommandBufferHeader *command_header,
        DeviceMoERebalanceWaveState *wave_state,
        DeviceMoERebalanceGraphControllerState *controller_state,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::runDeviceRebalanceController] invalid device rebalance config");
            return false;
        }
        if (!runtime_layers || !gathered_histograms || !status)
        {
            LOG_ERROR("[CUDAMoEKernel::runDeviceRebalanceController] runtime layers, gathered histograms, and status must be non-null");
            return false;
        }
        void *stream = requireStream("CUDAMoEKernel::runDeviceRebalanceController");
        if (!stream)
            return false;
        if (!setMoEDevice(device_ordinal_, "runDeviceRebalanceController"))
            return false;

        return cudaMoE_device_rebalance_controller(
            runtime_layers,
            reinterpret_cast<const unsigned long long *>(gathered_histograms),
            status,
            &config,
            plan_entries,
            plan_count,
            plan_capacity,
            payload_slot_capacity,
            command_header,
            wave_state,
            controller_state,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::packDeviceRebalanceHistograms(
        DeviceMoELayerRuntime *runtime_layers,
        uint64_t *local_histograms,
        const DeviceMoERebalanceConfig &config,
        const DeviceMoERebalanceWaveState *wave_state,
        const DeviceMoERebalanceGraphControllerState *controller_state,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::packDeviceRebalanceHistograms] invalid device rebalance config");
            return false;
        }
        if (!runtime_layers || !local_histograms)
        {
            LOG_ERROR("[CUDAMoEKernel::packDeviceRebalanceHistograms] runtime layers and local histograms must be non-null");
            return false;
        }
        void *stream = requireStream("CUDAMoEKernel::packDeviceRebalanceHistograms");
        if (!setMoEDevice(device_ordinal_, "packDeviceRebalanceHistograms"))
            return false;

        return cudaMoE_pack_rebalance_histograms(
            runtime_layers,
            reinterpret_cast<unsigned long long *>(local_histograms),
            &config,
            wave_state,
            controller_state,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::packDeviceRebalanceDirectory(
        DeviceMoELayerRuntime *runtime_layers,
        DeviceMoEExpertDirectoryEntry *local_directory,
        const DeviceMoERebalanceConfig &config)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::packDeviceRebalanceDirectory] invalid device rebalance config");
            return false;
        }
        if (!runtime_layers || !local_directory)
        {
            LOG_ERROR("[CUDAMoEKernel::packDeviceRebalanceDirectory] runtime layers and local directory must be non-null");
            return false;
        }
        void *stream = requireStream("CUDAMoEKernel::packDeviceRebalanceDirectory");
        if (!setMoEDevice(device_ordinal_, "packDeviceRebalanceDirectory"))
            return false;

        return cudaMoE_pack_rebalance_directory(
            runtime_layers,
            local_directory,
            &config,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::packDeviceRebalanceSourceDescriptors(
        DeviceMoELayerRuntime *runtime_layers,
        const DeviceMoERebalancePlanEntry *plan_entries,
        const DeviceMoERebalanceCommandBufferHeader *command_headers,
        uint32_t plan_capacity,
        DeviceMoEExpertDirectoryEntry *local_source_descriptors,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceGraphControllerState *controller_state,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::packDeviceRebalanceSourceDescriptors] invalid device rebalance config");
            return false;
        }
        if (!runtime_layers || !plan_entries || !command_headers ||
            !local_source_descriptors)
        {
            LOG_ERROR("[CUDAMoEKernel::packDeviceRebalanceSourceDescriptors] runtime, projected plans, headers, and source descriptor output must be non-null");
            return false;
        }
        void *stream = requireStream("CUDAMoEKernel::packDeviceRebalanceSourceDescriptors");
        if (!setMoEDevice(device_ordinal_, "packDeviceRebalanceSourceDescriptors"))
            return false;

        return cudaMoE_pack_rebalance_source_descriptors(
            runtime_layers,
            plan_entries,
            command_headers,
            plan_capacity,
            local_source_descriptors,
            &config,
            controller_state,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::projectDeviceRebalanceDomainCommands(
        const DeviceMoERebalancePlanEntry *gathered_plan_entries,
        const DeviceMoERebalanceCommandBufferHeader *gathered_command_headers,
        uint32_t plan_capacity,
        DeviceMoERebalancePlanEntry *local_plan_entries,
        DeviceMoERebalanceCommandBufferHeader *local_command_headers,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceStatus *status,
        uint32_t payload_slot_capacity,
        uint32_t command_buffer_count,
        const DeviceMoERebalanceWaveState *gathered_wave_states,
        DeviceMoERebalanceWaveState *local_wave_states)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::projectDeviceRebalanceDomainCommands] invalid device rebalance config");
            return false;
        }
        if (!gathered_plan_entries || !gathered_command_headers ||
            !local_plan_entries || !local_command_headers ||
            plan_capacity == 0)
        {
            LOG_ERROR("[CUDAMoEKernel::projectDeviceRebalanceDomainCommands] gathered and local command buffers must be non-null");
            return false;
        }
        void *stream = requireStream("CUDAMoEKernel::projectDeviceRebalanceDomainCommands");
        if (!setMoEDevice(device_ordinal_, "projectDeviceRebalanceDomainCommands"))
            return false;

        return cudaMoE_project_rebalance_domain_commands(
            gathered_plan_entries,
            gathered_command_headers,
            plan_capacity,
            local_plan_entries,
            local_command_headers,
            &config,
            status,
            payload_slot_capacity,
            command_buffer_count,
            gathered_wave_states,
            local_wave_states,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::projectPrefillLeastLoadedDomainCommands(
        const DeviceMoERebalancePlanEntry *gathered_plan_entries,
        const DeviceMoERebalanceCommandBufferHeader *gathered_command_headers,
        uint32_t plan_capacity,
        DeviceMoERebalancePlanEntry *local_plan_entries,
        uint32_t *local_plan_count,
        DeviceMoERebalanceCommandBufferHeader *local_command_header,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceStatus *status,
        uint32_t payload_slot_capacity,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::projectPrefillLeastLoadedDomainCommands] invalid device rebalance config");
            return false;
        }
        if (!gathered_plan_entries || !gathered_command_headers ||
            !local_plan_entries || !local_plan_count || !local_command_header ||
            !status || plan_capacity == 0 || payload_slot_capacity == 0)
        {
            LOG_ERROR("[CUDAMoEKernel::projectPrefillLeastLoadedDomainCommands] gathered commands, local command output, status, and payload capacity are required");
            return false;
        }
        void *stream = requireStream("CUDAMoEKernel::projectPrefillLeastLoadedDomainCommands");
        if (!setMoEDevice(device_ordinal_, "projectPrefillLeastLoadedDomainCommands"))
            return false;

        return cudaMoE_project_prefill_llep_domain_commands(
            gathered_plan_entries,
            gathered_command_headers,
            plan_capacity,
            local_plan_entries,
            local_plan_count,
            local_command_header,
            &config,
            status,
            payload_slot_capacity,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::materializePrefillLeastLoadedTransferCommands(
        const DeviceMoELayerRuntime *runtime_layer,
        DeviceMoERebalancePlanEntry *plan_entries,
        uint32_t *plan_count,
        uint32_t plan_capacity,
        DeviceMoERebalanceCommandBufferHeader *command_header,
        DeviceMoERebalanceStatus *status,
        const DeviceMoERebalanceConfig &config,
        uint32_t payload_slot_capacity,
        uint32_t layer_idx,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::materializePrefillLeastLoadedTransferCommands] invalid device rebalance config");
            return false;
        }
        if (!runtime_layer || !plan_entries || !plan_count || !command_header || !status ||
            plan_capacity == 0 || payload_slot_capacity == 0)
        {
            LOG_ERROR("[CUDAMoEKernel::materializePrefillLeastLoadedTransferCommands] runtime, command buffers, status, and payload capacity are required");
            return false;
        }
        if (layer_idx >= config.num_layers)
        {
            LOG_ERROR("[CUDAMoEKernel::materializePrefillLeastLoadedTransferCommands] layer index out of range"
                      << " layer=" << layer_idx << " num_layers=" << config.num_layers);
            return false;
        }
        void *stream = requireStream("CUDAMoEKernel::materializePrefillLeastLoadedTransferCommands");
        if (!setMoEDevice(device_ordinal_, "materializePrefillLeastLoadedTransferCommands"))
            return false;

        return cudaMoE_materialize_prefill_llep_transfer_commands(
            runtime_layer,
            plan_entries,
            plan_count,
            plan_capacity,
            command_header,
            status,
            &config,
            payload_slot_capacity,
            layer_idx,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::packDeviceRebalanceCompactPayloads(
        const DeviceMoERebalancePlanEntry *plan_entries,
        const DeviceMoERebalanceCommandBufferHeader *command_headers,
        uint32_t plan_capacity,
        const DeviceMoEExpertDirectoryEntry *local_source_descriptors,
        uint8_t *local_payload,
        uint32_t local_payload_slot_count,
        uint64_t payload_slot_bytes,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceApplyStatus *status,
        DeviceMoERebalanceGraphControllerState *controller_state,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::packDeviceRebalanceCompactPayloads] invalid device rebalance config");
            return false;
        }
        if (!plan_entries || !command_headers ||
            !local_source_descriptors || !local_payload || !status ||
            plan_capacity == 0 || local_payload_slot_count == 0 ||
            payload_slot_bytes <= sizeof(DeviceMoEExpertDirectoryEntry))
        {
            LOG_ERROR("[CUDAMoEKernel::packDeviceRebalanceCompactPayloads] plans, headers, source descriptors, payload, and status must be non-null");
            return false;
        }
        void *stream = requireStream("CUDAMoEKernel::packDeviceRebalanceCompactPayloads");
        if (!setMoEDevice(device_ordinal_, "packDeviceRebalanceCompactPayloads"))
            return false;

        return cudaMoE_pack_rebalance_compact_payloads(
            plan_entries,
            command_headers,
            plan_capacity,
            local_source_descriptors,
            local_payload,
            local_payload_slot_count,
            static_cast<unsigned long long>(payload_slot_bytes),
            &config,
            status,
            controller_state,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::applyDeviceRebalanceArrivals(
        DeviceMoELayerRuntime *runtime_layers,
        const DeviceMoERebalancePlanEntry *plan_entries,
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        const DeviceMoEExpertDirectoryEntry *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceApplyStatus *status,
        const DeviceMoERebalanceCommandBufferHeader *command_header,
        int target_layer)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::applyDeviceRebalanceArrivals] invalid device rebalance config");
            return false;
        }
        if (!runtime_layers || !plan_entries || (!plan_count && !command_header) ||
            !local_transfer_slots || !status)
        {
            LOG_ERROR("[CUDAMoEKernel::applyDeviceRebalanceArrivals] runtime, plan, transfer slots, and status must be non-null");
            return false;
        }
        void *stream = requireStream("CUDAMoEKernel::applyDeviceRebalanceArrivals");
        if (!setMoEDevice(device_ordinal_, "applyDeviceRebalanceArrivals"))
            return false;

        return cudaMoE_apply_rebalance_arrivals(
            runtime_layers,
            plan_entries,
            plan_count,
            plan_capacity,
            command_header,
            local_transfer_slots,
            local_transfer_slot_count,
            &config,
            status,
            target_layer,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::packDeviceRebalanceCollectivePayloads(
        const DeviceMoERebalancePlanEntry *gathered_plan_entries,
        const DeviceMoERebalanceCommandBufferHeader *gathered_command_headers,
        uint32_t plan_capacity,
        const DeviceMoEExpertDirectoryEntry *local_directory,
        uint8_t *local_payload,
        uint32_t local_payload_slot_count,
        uint64_t payload_slot_bytes,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceApplyStatus *status,
        DeviceMoERebalanceGraphControllerState *controller_state,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::packDeviceRebalanceCollectivePayloads] invalid device rebalance config");
            return false;
        }
        if (!gathered_plan_entries || !gathered_command_headers ||
            !local_directory || !local_payload || !status ||
            plan_capacity == 0 || local_payload_slot_count == 0 ||
            payload_slot_bytes == 0)
        {
            LOG_ERROR("[CUDAMoEKernel::packDeviceRebalanceCollectivePayloads] gathered plans, headers, local directory, payload, and status must be non-null");
            return false;
        }
        void *stream = requireStream("CUDAMoEKernel::packDeviceRebalanceCollectivePayloads");
        if (!setMoEDevice(device_ordinal_, "packDeviceRebalanceCollectivePayloads"))
            return false;

        return cudaMoE_pack_rebalance_collective_payloads(
            gathered_plan_entries,
            gathered_command_headers,
            plan_capacity,
            local_directory,
            local_payload,
            local_payload_slot_count,
            static_cast<unsigned long long>(payload_slot_bytes),
            &config,
            status,
            controller_state,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::unpackDeviceRebalanceCollectivePayloads(
        const DeviceMoERebalancePlanEntry *plan_entries,
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        const DeviceMoERebalanceCommandBufferHeader *command_header,
        const uint8_t *gathered_payload,
        uint32_t local_payload_slot_count,
        uint64_t payload_slot_bytes,
        DeviceMoEExpertDirectoryEntry *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceApplyStatus *status,
        DeviceMoERebalanceGraphControllerState *controller_state,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::unpackDeviceRebalanceCollectivePayloads] invalid device rebalance config");
            return false;
        }
        if (!plan_entries || (!plan_count && !command_header) ||
            !gathered_payload || !local_transfer_slots || !status ||
            plan_capacity == 0 || local_payload_slot_count == 0 ||
            payload_slot_bytes <= sizeof(DeviceMoEExpertDirectoryEntry))
        {
            LOG_ERROR("[CUDAMoEKernel::unpackDeviceRebalanceCollectivePayloads] plan, gathered payload, transfer slots, and status must be non-null");
            return false;
        }
        void *stream = requireStream("CUDAMoEKernel::unpackDeviceRebalanceCollectivePayloads");
        if (!setMoEDevice(device_ordinal_, "unpackDeviceRebalanceCollectivePayloads"))
            return false;

        return cudaMoE_unpack_rebalance_collective_payloads(
            plan_entries,
            plan_count,
            plan_capacity,
            command_header,
            gathered_payload,
            local_payload_slot_count,
            static_cast<unsigned long long>(payload_slot_bytes),
            local_transfer_slots,
            local_transfer_slot_count,
            &config,
            status,
            controller_state,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::initializeDeviceRebalanceGraphController(
        DeviceMoERebalanceGraphControllerState *controller_state,
        const DeviceMoERebalanceConfig &config)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::initializeDeviceRebalanceGraphController] invalid device rebalance config");
            return false;
        }
        if (!controller_state)
        {
            LOG_ERROR("[CUDAMoEKernel::initializeDeviceRebalanceGraphController] controller state must be non-null");
            return false;
        }
        void *stream = requireStream("CUDAMoEKernel::initializeDeviceRebalanceGraphController");
        if (!stream)
            return false;
        if (!setMoEDevice(device_ordinal_, "initializeDeviceRebalanceGraphController"))
            return false;

        return cudaMoE_init_rebalance_graph_controller_state(
            controller_state,
            &config,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::publishDeviceRebalanceTransferComplete(
        DeviceMoERebalanceGraphControllerState *controller_state,
        const DeviceMoERebalanceCommandBufferHeader *command_header,
        const DeviceMoERebalanceWaveState *wave_state,
        const DeviceMoERebalanceApplyStatus *copy_status,
        const DeviceMoERebalancePlanEntry *plan_entries,
        uint32_t plan_capacity,
        const DeviceMoERebalanceApplyStatus *gathered_copy_status,
        const DeviceMoERebalanceConfig &config,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::publishDeviceRebalanceTransferComplete] invalid device rebalance config");
            return false;
        }
        if (!controller_state || !command_header || !wave_state || !copy_status ||
            !plan_entries || plan_capacity == 0u || !gathered_copy_status)
        {
            LOG_ERROR("[CUDAMoEKernel::publishDeviceRebalanceTransferComplete] controller state, command header, wave state, copy status, plan entries, and gathered copy status must be non-null");
            return false;
        }
        void *stream = requireStream("CUDAMoEKernel::publishDeviceRebalanceTransferComplete");
        if (!stream)
            return false;
        if (!setMoEDevice(device_ordinal_, "publishDeviceRebalanceTransferComplete"))
            return false;

        return cudaMoE_publish_rebalance_transfer_complete(
            controller_state,
            command_header,
            wave_state,
            copy_status,
            plan_entries,
            plan_capacity,
            gathered_copy_status,
            &config,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::applyReadyDeviceRebalanceWave(
        DeviceMoELayerRuntime *runtime_layers,
        const DeviceMoERebalancePlanEntry *plan_entries,
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        const DeviceMoEExpertDirectoryEntry *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceApplyStatus *status,
        DeviceMoERebalanceGraphControllerState *controller_state,
        DeviceMoERebalanceCommandBufferHeader *command_header,
        int target_layer,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[CUDAMoEKernel::applyReadyDeviceRebalanceWave] invalid device rebalance config");
            return false;
        }
        if (!runtime_layers || !plan_entries || (!plan_count && !command_header) ||
            !status || !controller_state)
        {
            LOG_ERROR("[CUDAMoEKernel::applyReadyDeviceRebalanceWave] runtime, plan, status, controller state, and command ABI must be non-null");
            return false;
        }
        void *stream = requireStream("CUDAMoEKernel::applyReadyDeviceRebalanceWave");
        if (!stream)
            return false;
        if (!setMoEDevice(device_ordinal_, "applyReadyDeviceRebalanceWave"))
            return false;

        return cudaMoE_apply_ready_rebalance_wave(
            runtime_layers,
            plan_entries,
            plan_count,
            plan_capacity,
            command_header,
            local_transfer_slots,
            local_transfer_slot_count,
            &config,
            status,
            controller_state,
            target_layer,
            command_buffer_count,
            device_ordinal_,
            stream);
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

    bool CUDAMoEKernel::copyTokenRowFromTensor(
        ITensor *source, ITensor *row_buffer,
        int row_index, int row_width)
    {
        if (row_index < 0 || row_width <= 0)
            return false;
        void *stream = requireStream("CUDAMoEKernel::copyTokenRowFromTensor");
        const DeviceId device = deviceId();
        if (!ensureTensorOnDevice(source, device, stream, "source") ||
            !ensureOutputOnDevice(row_buffer, device, stream, "row_buffer"))
        {
            return false;
        }

        const auto *src = static_cast<const float *>(source->gpu_data_ptr());
        auto *dst = static_cast<float *>(row_buffer->gpu_data_ptr());
        if (!cudaMoE_copy_token_row(src, dst, row_index, row_width, device.ordinal, stream))
            return false;

        markDeviceWritten(row_buffer, device, stream);
        return true;
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

    bool CUDAMoEKernel::writeTokenRowToTensor(
        ITensor *destination, ITensor *row_buffer,
        int row_index, int row_width)
    {
        if (row_index < 0 || row_width <= 0)
            return false;
        void *stream = requireStream("CUDAMoEKernel::writeTokenRowToTensor");
        const DeviceId device = deviceId();
        if (!ensureOutputOnDevice(destination, device, stream, "destination") ||
            !ensureTensorOnDevice(row_buffer, device, stream, "row_buffer"))
        {
            return false;
        }

        auto *dst = static_cast<float *>(destination->gpu_data_ptr());
        const auto *src = static_cast<const float *>(row_buffer->gpu_data_ptr());
        if (!cudaMoE_write_token_row(dst, src, row_index, row_width, device.ordinal, stream))
            return false;

        markDeviceWritten(destination, device, stream);
        return true;
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

    bool CUDAMoEKernel::sharedExpertGateFromTensorsEffectiveSeqLen(
        ITensor *input, ITensor *gate_inp, ITensor *shared_output,
        int seq_len, int d_model,
        const int *device_effective_seq_len)
    {
        if (seq_len <= 0)
            return true;
        if (!device_effective_seq_len)
        {
            LOG_ERROR("[CUDAMoEKernel::sharedExpertGateFromTensorsEffectiveSeqLen] missing device effective length scalar");
            return false;
        }

        void *stream = requireStream("CUDAMoEKernel::sharedExpertGateFromTensorsEffectiveSeqLen");
        const DeviceId device = deviceId();
        if (!ensureTensorOnDevice(input, device, stream, "input") ||
            !ensureTensorOnDevice(gate_inp, device, stream, "gate_inp") ||
            !ensureTensorOnDevice(shared_output, device, stream, "shared_output"))
            return false;

        const auto *in = static_cast<const float *>(input->gpu_data_ptr());
        const auto *gi = static_cast<const float *>(gate_inp->gpu_data_ptr());
        auto *so = static_cast<float *>(shared_output->gpu_data_ptr());
        if (!in || !gi || !so)
        {
            LOG_ERROR("[CUDAMoEKernel::sharedExpertGateFromTensorsEffectiveSeqLen] null device pointer");
            return false;
        }

        if (!cudaMoE_shared_expert_gate_effective_seq_len(
                in, gi, so, seq_len, d_model, device_effective_seq_len,
                device_ordinal_, stream))
        {
            LOG_ERROR("[CUDAMoEKernel::sharedExpertGateFromTensorsEffectiveSeqLen] kernel launch failed");
            return false;
        }
        markDeviceWritten(shared_output, device, stream);
        return true;
    }

    void CUDAMoEKernel::sharedExpertGateAddFromTensors(
        ITensor *input, ITensor *gate_inp, ITensor *shared_output,
        ITensor *routed_residual, ITensor *combined_output,
        int seq_len, int d_model)
    {
        if (seq_len <= 0)
            return;

        void *stream = requireStream("CUDAMoEKernel::sharedExpertGateAddFromTensors");
        const DeviceId device = deviceId();
        if (!ensureTensorOnDevice(input, device, stream, "input") ||
            !ensureTensorOnDevice(gate_inp, device, stream, "gate_inp") ||
            !ensureTensorOnDevice(shared_output, device, stream, "shared_output") ||
            !ensureTensorOnDevice(routed_residual, device, stream, "routed_residual") ||
            !ensureOutputOnDevice(combined_output, device, stream, "combined_output"))
            return;

        const auto *in = static_cast<const float *>(input->gpu_data_ptr());
        const auto *gi = static_cast<const float *>(gate_inp->gpu_data_ptr());
        auto *so = static_cast<float *>(shared_output->gpu_data_ptr());
        const auto *rr = static_cast<const float *>(routed_residual->gpu_data_ptr());
        auto *co = static_cast<float *>(combined_output->gpu_data_ptr());
        if (!in || !gi || !so || !rr || !co)
        {
            LOG_ERROR("[CUDAMoEKernel::sharedExpertGateAddFromTensors] null device pointer");
            return;
        }

        if (!cudaMoE_shared_expert_gate_add(in, gi, so, rr, co,
                                            seq_len, d_model, device_ordinal_, stream))
        {
            LOG_ERROR("[CUDAMoEKernel::sharedExpertGateAddFromTensors] fused gate-add kernel launch failed");
            return;
        }
        markDeviceWritten(shared_output, device, stream);
        markDeviceWritten(combined_output, device, stream);
    }

    bool CUDAMoEKernel::sharedExpertGateAddFromTensorsEffectiveSeqLen(
        ITensor *input, ITensor *gate_inp, ITensor *shared_output,
        ITensor *routed_residual, ITensor *combined_output,
        int seq_len, int d_model,
        const int *device_effective_seq_len)
    {
        if (seq_len <= 0)
            return true;
        if (!device_effective_seq_len)
        {
            LOG_ERROR("[CUDAMoEKernel::sharedExpertGateAddFromTensorsEffectiveSeqLen] missing device effective length scalar");
            return false;
        }

        void *stream = requireStream("CUDAMoEKernel::sharedExpertGateAddFromTensorsEffectiveSeqLen");
        const DeviceId device = deviceId();
        if (!ensureTensorOnDevice(input, device, stream, "input") ||
            !ensureTensorOnDevice(gate_inp, device, stream, "gate_inp") ||
            !ensureTensorOnDevice(shared_output, device, stream, "shared_output") ||
            !ensureTensorOnDevice(routed_residual, device, stream, "routed_residual") ||
            !ensureOutputOnDevice(combined_output, device, stream, "combined_output"))
            return false;

        const auto *in = static_cast<const float *>(input->gpu_data_ptr());
        const auto *gi = static_cast<const float *>(gate_inp->gpu_data_ptr());
        auto *so = static_cast<float *>(shared_output->gpu_data_ptr());
        const auto *rr = static_cast<const float *>(routed_residual->gpu_data_ptr());
        auto *co = static_cast<float *>(combined_output->gpu_data_ptr());
        if (!in || !gi || !so || !rr || !co)
        {
            LOG_ERROR("[CUDAMoEKernel::sharedExpertGateAddFromTensorsEffectiveSeqLen] null device pointer");
            return false;
        }

        if (!cudaMoE_shared_expert_gate_add_effective_seq_len(
                in, gi, so, rr, co, seq_len, d_model,
                device_effective_seq_len, device_ordinal_, stream))
        {
            LOG_ERROR("[CUDAMoEKernel::sharedExpertGateAddFromTensorsEffectiveSeqLen] fused gate-add kernel launch failed");
            return false;
        }
        markDeviceWritten(shared_output, device, stream);
        markDeviceWritten(combined_output, device, stream);
        return true;
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
        /*
         * Padded prefill replay deliberately marks bucket-tail routes invalid
         * with expert_id=-1.  The deterministic scatter kernel only writes a
         * mapping for valid routes, so the ordered down-scatter must start from
         * an all-invalid map every request.  Otherwise graph replay can reuse
         * stale slot mappings from the previous real sequence length and write
         * arbitrary expert output into padded rows.
         */
        err = cudaMemsetAsync(d_group_original_to_grouped_, 0xff,
                              static_cast<size_t>(total_slots) * sizeof(int),
                              static_cast<cudaStream_t>(stream));
        if (err != cudaSuccess)
            return false;
        err = cudaMemsetAsync(d_group_original_expert_ids_, 0xff,
                              static_cast<size_t>(total_slots) * sizeof(int),
                              static_cast<cudaStream_t>(stream));
        if (err != cudaSuccess)
            return false;
        /*
         * Masked/padded routing can compact fewer rows than total_slots.  The
         * grouped prefill gather still launches over total_slots, so stale token
         * ids from a previous larger request must not survive in unused rows.
         */
        err = cudaMemsetAsync(d_grouped_token_indices, 0,
                              static_cast<size_t>(total_slots) * sizeof(int),
                              static_cast<cudaStream_t>(stream));
        if (err != cudaSuccess)
            return false;
        err = cudaMemsetAsync(d_grouped_weights, 0,
                              static_cast<size_t>(total_slots) * sizeof(float),
                              static_cast<cudaStream_t>(stream));
        if (err != cudaSuccess)
            return false;
        return cudaMoE_count_per_expert(d_routing_indices, d_expert_counts, total_slots,
                                        num_experts, device_ordinal_, stream) &&
               cudaMoE_exclusive_scan(d_expert_counts, d_expert_offsets,
                                       num_experts, device_ordinal_, stream) &&
               cudaMoE_build_active_expert_list(
                   d_expert_counts, d_group_active_expert_ids_,
                   num_experts, std::min(total_slots, num_experts),
                   device_ordinal_, stream) &&
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

    bool CUDAMoEKernel::groupPrefillRoutes(
        DeviceMoELayerRuntime *runtime_layer,
        ITensor *routing_indices,
        ITensor *routing_weights,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k,
        bool filter_to_local_runtime_experts)
    {
        if (!runtime_layer || !routing_indices || !routing_weights)
        {
            LOG_ERROR("[CUDAMoEKernel::groupPrefillRoutes] null runtime or routing tensor");
            return false;
        }
        if (current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0)
        {
            LOG_ERROR("[CUDAMoEKernel::groupPrefillRoutes] invalid dimensions current_tokens="
                      << current_tokens << " max_tokens=" << max_tokens
                      << " num_experts=" << num_experts << " top_k=" << top_k);
            return false;
        }

        void *stream = requireStream("CUDAMoEKernel::groupPrefillRoutes");
        const DeviceId device = deviceId();
        if (!ensureTensorOnDevice(routing_indices, device, stream, "routing_indices") ||
            !ensureTensorOnDevice(routing_weights, device, stream, "routing_weights"))
        {
            return false;
        }

        const float *d_indices = static_cast<const float *>(routing_indices->gpu_data_ptr());
        const float *d_weights = static_cast<const float *>(routing_weights->gpu_data_ptr());
        if (!d_indices || !d_weights)
        {
            LOG_ERROR("[CUDAMoEKernel::groupPrefillRoutes] routing tensors have no device pointers");
            return false;
        }

        if (traceMoEPrefillGroupingWrapperEnabled())
        {
            LOG_INFO("[CUDAMoEKernel] prefill grouping wrapper trace"
                     << " tag=groupPrefillRoutes"
                     << " device=" << device.to_string()
                     << " current_tokens=" << current_tokens
                     << " max_tokens=" << max_tokens
                     << " num_experts=" << num_experts
                     << " top_k=" << top_k
                     << " runtime_layer=" << static_cast<void *>(runtime_layer)
                     << " routing_indices=" << static_cast<const void *>(d_indices)
                     << " routing_weights=" << static_cast<const void *>(d_weights));
        }

        return cudaMoE_group_prefill_routes_runtime(
            d_indices,
            d_weights,
            static_cast<void *>(runtime_layer),
            current_tokens * top_k,
            max_tokens * top_k,
            num_experts,
            top_k,
            filter_to_local_runtime_experts ? 1 : 0,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::regroupPrefillRoutesFromRuntimeAssignments(
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k)
    {
        if (!runtime_layer)
        {
            LOG_ERROR("[CUDAMoEKernel::regroupPrefillRoutesFromRuntimeAssignments] null runtime");
            return false;
        }
        if (current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0)
        {
            LOG_ERROR("[CUDAMoEKernel::regroupPrefillRoutesFromRuntimeAssignments] invalid dimensions current_tokens="
                      << current_tokens << " max_tokens=" << max_tokens
                      << " num_experts=" << num_experts << " top_k=" << top_k);
            return false;
        }
        void *stream = requireStream("CUDAMoEKernel::regroupPrefillRoutesFromRuntimeAssignments");
        return cudaMoE_regroup_prefill_routes_runtime_assignments(
            static_cast<void *>(runtime_layer),
            current_tokens * top_k,
            max_tokens * top_k,
            num_experts,
            top_k,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::assignPrefillRoutesLeastLoadedResident(
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k)
    {
        if (!runtime_layer)
        {
            LOG_ERROR("[CUDAMoEKernel::assignPrefillRoutesLeastLoadedResident] null runtime");
            return false;
        }
        if (current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0)
        {
            LOG_ERROR("[CUDAMoEKernel::assignPrefillRoutesLeastLoadedResident] invalid dimensions current_tokens="
                      << current_tokens << " max_tokens=" << max_tokens
                      << " num_experts=" << num_experts << " top_k=" << top_k);
            return false;
        }
        void *stream = requireStream("CUDAMoEKernel::assignPrefillRoutesLeastLoadedResident");
        return cudaMoE_assign_prefill_routes_least_loaded_resident(
            static_cast<void *>(runtime_layer),
            current_tokens * top_k,
            max_tokens * top_k,
            num_experts,
            top_k,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::planPrefillRoutesLeastLoadedCurrentBatch(
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k,
        const least_loaded_ep::LeastLoadedExpertAssignmentConfig &config)
    {
        if (!runtime_layer)
        {
            LOG_ERROR("[CUDAMoEKernel::planPrefillRoutesLeastLoadedCurrentBatch] null runtime");
            return false;
        }
        if (current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0)
        {
            LOG_ERROR("[CUDAMoEKernel::planPrefillRoutesLeastLoadedCurrentBatch] invalid dimensions current_tokens="
                      << current_tokens << " max_tokens=" << max_tokens
                      << " num_experts=" << num_experts << " top_k=" << top_k);
            return false;
        }
        if (config.expert_count != static_cast<uint32_t>(num_experts) ||
            config.participant_count == 0u ||
            config.alpha_numerator == 0u ||
            config.alpha_denominator == 0u ||
            config.lambda_numerator == 0u ||
            config.lambda_denominator == 0u)
        {
            LOG_ERROR("[CUDAMoEKernel::planPrefillRoutesLeastLoadedCurrentBatch] invalid LLEP config");
            return false;
        }

        void *stream = requireStream("CUDAMoEKernel::planPrefillRoutesLeastLoadedCurrentBatch");
        return cudaMoE_plan_prefill_routes_least_loaded_current_batch(
            static_cast<void *>(runtime_layer),
            current_tokens * top_k,
            max_tokens * top_k,
            num_experts,
            top_k,
            config.min_chunk_tokens,
            config.alpha_numerator,
            config.alpha_denominator,
            config.lambda_numerator,
            config.lambda_denominator,
            config.min_spread_improvement,
            config.min_spread_improvement_divisor,
            config.min_spread_improvement_per_transfer,
            config.min_foreign_rows_per_transfer,
            config.max_weight_transfers,
            config.enable_balanced_skip ? 1 : 0,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers(
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k)
    {
        if (!runtime_layer)
        {
            LOG_ERROR("[CUDAMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers] null runtime");
            return false;
        }
        if (current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0)
        {
            LOG_ERROR("[CUDAMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers] invalid dimensions current_tokens="
                      << current_tokens << " max_tokens=" << max_tokens
                      << " num_experts=" << num_experts << " top_k=" << top_k);
            return false;
        }

        void *stream = requireStream(
            "CUDAMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers");
        return cudaMoE_assign_prefill_routes_from_llep_current_batch_plan_no_transfers(
            static_cast<void *>(runtime_layer),
            current_tokens * top_k,
            max_tokens * top_k,
            num_experts,
            top_k,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers(
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens,
        int max_tokens,
        int num_experts,
        int top_k,
        const DeviceMoERebalanceStatus *transfer_status,
        const DeviceMoERebalanceApplyStatus *apply_status)
    {
        if (!runtime_layer)
        {
            LOG_ERROR("[CUDAMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers] null runtime");
            return false;
        }
        if (!transfer_status || !apply_status)
        {
            LOG_ERROR("[CUDAMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers] transfer and apply status are required");
            return false;
        }
        if (current_tokens < 0 || max_tokens <= 0 || current_tokens > max_tokens ||
            num_experts <= 0 || top_k <= 0)
        {
            LOG_ERROR("[CUDAMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers] invalid dimensions current_tokens="
                      << current_tokens << " max_tokens=" << max_tokens
                      << " num_experts=" << num_experts << " top_k=" << top_k);
            return false;
        }

        void *stream = requireStream(
            "CUDAMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers");
        if (!requireCudaDevicePointer(
                runtime_layer,
                device_ordinal_,
                "runtime_layer",
                "assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers",
                stream) ||
            !requireCudaDevicePointer(
                transfer_status,
                device_ordinal_,
                "transfer_status",
                "assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers",
                stream) ||
            !requireCudaDevicePointer(
                apply_status,
                device_ordinal_,
                "apply_status",
                "assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers",
                stream))
        {
            return false;
        }
        return cudaMoE_assign_prefill_routes_from_llep_current_batch_plan_after_transfers(
            static_cast<void *>(runtime_layer),
            current_tokens * top_k,
            max_tokens * top_k,
            num_experts,
            top_k,
            transfer_status,
            apply_status,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::gatherPrefillExpertBatchFromRuntime(
        DeviceMoELayerRuntime *runtime_layer,
        ITensor *hidden,
        ITensor *batch_buffer,
        int expert_id,
        int max_tokens,
        int d_model)
    {
        if (!runtime_layer || !hidden || !batch_buffer)
        {
            LOG_ERROR("[CUDAMoEKernel::gatherPrefillExpertBatchFromRuntime] null runtime/input/output tensor");
            return false;
        }
        if (expert_id < 0 || max_tokens <= 0 || d_model <= 0)
        {
            LOG_ERROR("[CUDAMoEKernel::gatherPrefillExpertBatchFromRuntime] invalid arguments expert_id="
                      << expert_id << " max_tokens=" << max_tokens << " d_model=" << d_model);
            return false;
        }

        void *stream = requireStream("CUDAMoEKernel::gatherPrefillExpertBatchFromRuntime");
        const DeviceId device = deviceId();
        if (!ensureTensorOnDevice(hidden, device, stream, "hidden") ||
            !ensureOutputOnDevice(batch_buffer, device, stream, "batch_buffer"))
        {
            return false;
        }

        const float *d_hidden = static_cast<const float *>(hidden->gpu_data_ptr());
        float *d_batch = static_cast<float *>(batch_buffer->gpu_data_ptr());
        if (!d_hidden || !d_batch)
        {
            LOG_ERROR("[CUDAMoEKernel::gatherPrefillExpertBatchFromRuntime] tensors have no device pointers");
            return false;
        }

        const bool ok = cudaMoE_prefill_gather_expert_runtime(
            static_cast<const void *>(runtime_layer),
            d_hidden,
            d_batch,
            expert_id,
            max_tokens,
            d_model,
            device_ordinal_,
            stream);
        if (ok)
            markDeviceWritten(batch_buffer, device, stream);
        return ok;
    }

    bool CUDAMoEKernel::scatterPrefillExpertResultsFromRuntime(
        ITensor *output,
        ITensor *expert_results,
        DeviceMoELayerRuntime *runtime_layer,
        int expert_id,
        int max_tokens,
        int d_model)
    {
        if (!output || !expert_results || !runtime_layer)
        {
            LOG_ERROR("[CUDAMoEKernel::scatterPrefillExpertResultsFromRuntime] null runtime/input/output tensor");
            return false;
        }
        if (expert_id < 0 || max_tokens <= 0 || d_model <= 0)
        {
            LOG_ERROR("[CUDAMoEKernel::scatterPrefillExpertResultsFromRuntime] invalid arguments expert_id="
                      << expert_id << " max_tokens=" << max_tokens << " d_model=" << d_model);
            return false;
        }

        void *stream = requireStream("CUDAMoEKernel::scatterPrefillExpertResultsFromRuntime");
        const DeviceId device = deviceId();
        if (!ensureOutputOnDevice(output, device, stream, "output") ||
            !ensureTensorOnDevice(expert_results, device, stream, "expert_results"))
        {
            return false;
        }

        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        const float *d_expert_output = static_cast<const float *>(expert_results->gpu_data_ptr());
        if (!d_output || !d_expert_output)
        {
            LOG_ERROR("[CUDAMoEKernel::scatterPrefillExpertResultsFromRuntime] tensors have no device pointers");
            return false;
        }

        const bool ok = cudaMoE_prefill_scatter_expert_runtime(
            d_output,
            d_expert_output,
            static_cast<const void *>(runtime_layer),
            expert_id,
            max_tokens,
            d_model,
            device_ordinal_,
            stream);
        if (ok)
            markDeviceWritten(output, device, stream);
        return ok;
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
        if (traceMoEPrefillGroupingWrapperEnabled())
        {
            uint64_t total_routes = 0;
            int max_count = 0;
            int nonzero_experts = 0;
            for (int count : host_expert_counts_)
            {
                if (count > 0)
                {
                    total_routes += static_cast<uint64_t>(count);
                    max_count = std::max(max_count, count);
                    ++nonzero_experts;
                }
            }
            LOG_INFO("[CUDAMoEKernel] prefill grouping wrapper trace"
                     << " tag=prepareExpertGroups"
                     << " device=" << device.to_string()
                     << " seq_len=" << seq_len
                     << " num_experts=" << num_experts
                     << " top_k=" << top_k
                     << " total_routes=" << total_routes
                     << " nonzero_experts=" << nonzero_experts
                     << " max_count=" << max_count
                     << " counts_hash=" << hashMoETraceVector(host_expert_counts_)
                     << " offsets_hash=" << hashMoETraceVector(host_expert_offsets_));
        }
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
        uint32_t codebook_mask = 0;
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &desc = down_descs[expert_id];
            if (isBlankGroupedDesc(desc))
                continue;

            if (!validateCudaGroupedDescShape(desc, d_model, intermediate))
            {
                LOG_DEBUG("[CUDAMoEKernel::uploadGroupedExpertDownDescriptorTable] Invalid descriptor for expert "
                          << expert_id);
                return -1;
            }
            codebook_mask |= cudaGroupedPrefillCodebookBit(desc.codebook_id);
            if (codebook_id == 0)
            {
                codebook_id = desc.codebook_id;
            }
        }
        if (codebook_mask == 0)
            return -1;
        if (codebook_mask & (codebook_mask - 1u))
            codebook_id = kCudaMoEMixedCodebookSentinel;

        if (cudaGroupedPrefillMaskNeedsIQTables(codebook_mask))
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

        /*
         * Descriptor tables are weight-shaped, not request-shaped.  Graph
         * recapture can ask for the same table repeatedly; returning the
         * existing id keeps CUDA aligned with ROCm's deterministic slot model
         * and prevents pointer-array workspace slots from growing with replay
         * attempts.
         */
        const size_t desc_bytes =
            static_cast<size_t>(num_experts) * sizeof(DeviceNativeVNNIMatrixDesc);
        for (size_t index = 0; index < grouped_down_desc_tables_.size(); ++index)
        {
            const auto &existing = grouped_down_desc_tables_[index];
            if (!existing.valid ||
                !existing.device_descs ||
                existing.num_experts != num_experts ||
                existing.d_model != d_model ||
                existing.intermediate != intermediate ||
                existing.codebook_id != codebook_id ||
                existing.codebook_mask != codebook_mask ||
                existing.host_descs.size() != static_cast<size_t>(num_experts))
            {
                continue;
            }
            if (std::memcmp(existing.host_descs.data(), down_descs, desc_bytes) == 0)
            {
                return static_cast<int>(index);
            }
        }

        GroupedDownDescriptorTable table;
        table.host_descs.assign(down_descs, down_descs + num_experts);
        table.num_experts = num_experts;
        table.d_model = d_model;
        table.intermediate = intermediate;
        table.codebook_id = codebook_id;
        table.codebook_mask = codebook_mask;

        const std::size_t slot = next_grouped_down_desc_workspace_slot_;
        if (slot >= static_cast<std::size_t>(MoEWorkspaceBuffers::kGroupedDescriptorTableSlots))
        {
            LOG_ERROR("[CUDAMoEKernel::uploadGroupedExpertDownDescriptorTable] descriptor workspace slots exhausted: "
                      << slot << " capacity=" << MoEWorkspaceBuffers::kGroupedDescriptorTableSlots);
            return -1;
        }
        void *device_desc_base = nullptr;
        const size_t table_bytes =
            static_cast<size_t>(MoEWorkspaceBuffers::kGroupedDescriptorTableSlots) * desc_bytes;
        cudaError_t err = cudaSuccess;
        if (!bindWorkspaceBuffer(&device_desc_base,
                                 MoEWorkspaceBuffers::CUDA_GROUPED_DOWN_DESC_TABLES,
                                 table_bytes,
                                 "CUDA grouped down descriptor tables"))
        {
            err = cudaErrorInvalidValue;
        }
        DeviceNativeVNNIMatrixDesc *device_descs =
            err == cudaSuccess
                ? static_cast<DeviceNativeVNNIMatrixDesc *>(device_desc_base) +
                      slot * static_cast<std::size_t>(num_experts)
                : nullptr;
        if (err == cudaSuccess)
            err = cudaMemcpyAsync(device_descs, table.host_descs.data(),
                                  desc_bytes,
                                  cudaMemcpyHostToDevice, static_cast<cudaStream_t>(getStream()));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel::uploadGroupedExpertDownDescriptorTable] descriptor upload failed: "
                      << cudaGetErrorString(err));
            return -1;
        }

        table.device_descs = device_descs;
        table.workspace_slot = slot;
        table.valid = true;
        grouped_down_desc_tables_.push_back(std::move(table));
        next_grouped_down_desc_workspace_slot_ = slot + 1;
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
        uint32_t codebook_mask = 0;
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &gate_desc = gate_descs[expert_id];
            const auto &up_desc = up_descs[expert_id];
            const bool gate_blank = isBlankGroupedDesc(gate_desc);
            const bool up_blank = isBlankGroupedDesc(up_desc);
            if (gate_blank || up_blank)
            {
                if (gate_blank && up_blank)
                    continue;
                LOG_DEBUG("[CUDAMoEKernel::uploadGroupedExpertGateUpDescriptorTables] Incomplete sparse descriptor pair for expert "
                          << expert_id);
                return -1;
            }

            if (!gate_desc.valid() || !up_desc.valid() ||
                gate_desc.codebook_id != up_desc.codebook_id ||
                !validateCudaGroupedDescShape(gate_desc, intermediate, d_model) ||
                !validateCudaGroupedDescShape(up_desc, intermediate, d_model))
            {
                LOG_DEBUG("[CUDAMoEKernel::uploadGroupedExpertGateUpDescriptorTables] Invalid gate/up descriptor pair for expert "
                          << expert_id);
                return -1;
            }
            codebook_mask |= cudaGroupedPrefillCodebookBit(gate_desc.codebook_id);
            if (codebook_id == 0)
            {
                codebook_id = gate_desc.codebook_id;
            }
        }
        if (codebook_mask == 0)
            return -1;
        if (codebook_mask & (codebook_mask - 1u))
            codebook_id = kCudaMoEMixedCodebookSentinel;

        if (cudaGroupedPrefillMaskNeedsIQTables(codebook_mask))
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

        /*
         * Gate/up descriptor tables are persistent prepared-weight metadata.
         * Reusing identical tables gives CUDA the same deterministic descriptor
         * id behavior as ROCm and keeps graph-owned pointer slots stable across
         * replay-state resets.
         */
        const size_t desc_bytes =
            static_cast<size_t>(num_experts) * sizeof(DeviceNativeVNNIMatrixDesc);
        for (size_t index = 0; index < grouped_gateup_desc_tables_.size(); ++index)
        {
            const auto &existing = grouped_gateup_desc_tables_[index];
            if (!existing.valid ||
                !existing.device_gate_descs ||
                !existing.device_up_descs ||
                existing.num_experts != num_experts ||
                existing.d_model != d_model ||
                existing.intermediate != intermediate ||
                existing.codebook_id != codebook_id ||
                existing.codebook_mask != codebook_mask ||
                existing.host_gate_descs.size() != static_cast<size_t>(num_experts) ||
                existing.host_up_descs.size() != static_cast<size_t>(num_experts))
            {
                continue;
            }
            if (std::memcmp(existing.host_gate_descs.data(), gate_descs, desc_bytes) == 0 &&
                std::memcmp(existing.host_up_descs.data(), up_descs, desc_bytes) == 0)
            {
                return static_cast<int>(index);
            }
        }

        GroupedGateUpDescriptorTable table;
        table.host_gate_descs.assign(gate_descs, gate_descs + num_experts);
        table.host_up_descs.assign(up_descs, up_descs + num_experts);
        table.num_experts = num_experts;
        table.d_model = d_model;
        table.intermediate = intermediate;
        table.codebook_id = codebook_id;
        table.codebook_mask = codebook_mask;

        const std::size_t slot = next_grouped_gateup_desc_workspace_slot_;
        if (slot >= static_cast<std::size_t>(MoEWorkspaceBuffers::kGroupedDescriptorTableSlots))
        {
            LOG_ERROR("[CUDAMoEKernel::uploadGroupedExpertGateUpDescriptorTables] descriptor workspace slots exhausted: "
                      << slot << " capacity=" << MoEWorkspaceBuffers::kGroupedDescriptorTableSlots);
            return -1;
        }
        void *device_gate_desc_base = nullptr;
        void *device_up_desc_base = nullptr;
        const size_t table_bytes =
            static_cast<size_t>(MoEWorkspaceBuffers::kGroupedDescriptorTableSlots) * desc_bytes;
        cudaError_t err = cudaSuccess;
        if (!bindWorkspaceBuffer(&device_gate_desc_base,
                                 MoEWorkspaceBuffers::CUDA_GROUPED_GATE_DESC_TABLES,
                                 table_bytes,
                                 "CUDA grouped gate descriptor tables") ||
            !bindWorkspaceBuffer(&device_up_desc_base,
                                 MoEWorkspaceBuffers::CUDA_GROUPED_UP_DESC_TABLES,
                                 table_bytes,
                                 "CUDA grouped up descriptor tables"))
        {
            err = cudaErrorInvalidValue;
        }
        DeviceNativeVNNIMatrixDesc *device_gate_descs =
            err == cudaSuccess
                ? static_cast<DeviceNativeVNNIMatrixDesc *>(device_gate_desc_base) +
                      slot * static_cast<std::size_t>(num_experts)
                : nullptr;
        DeviceNativeVNNIMatrixDesc *device_up_descs =
            err == cudaSuccess
                ? static_cast<DeviceNativeVNNIMatrixDesc *>(device_up_desc_base) +
                      slot * static_cast<std::size_t>(num_experts)
                : nullptr;
        if (err == cudaSuccess)
            err = cudaMemcpyAsync(device_gate_descs, table.host_gate_descs.data(),
                                  desc_bytes,
                                  cudaMemcpyHostToDevice, static_cast<cudaStream_t>(getStream()));
        if (err == cudaSuccess)
            err = cudaMemcpyAsync(device_up_descs, table.host_up_descs.data(),
                                  desc_bytes,
                                  cudaMemcpyHostToDevice, static_cast<cudaStream_t>(getStream()));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel::uploadGroupedExpertGateUpDescriptorTables] descriptor upload failed: "
                      << cudaGetErrorString(err));
            return -1;
        }

        table.device_gate_descs = device_gate_descs;
        table.device_up_descs = device_up_descs;
        table.workspace_slot = slot;
        table.valid = true;
        grouped_gateup_desc_tables_.push_back(std::move(table));
        next_grouped_gateup_desc_workspace_slot_ = slot + 1;
        return static_cast<int>(grouped_gateup_desc_tables_.size() - 1);
    }

    bool CUDAMoEKernel::updateGroupedExpertDownDescriptorTable(
        int descriptor_table_id,
        const DeviceNativeVNNIMatrixDesc *down_descs,
        int num_experts,
        int d_model,
        int intermediate)
    {
        if (!down_descs || descriptor_table_id < 0 ||
            descriptor_table_id >= static_cast<int>(grouped_down_desc_tables_.size()) ||
            num_experts <= 0 || d_model <= 0 || intermediate <= 0 ||
            (intermediate % 32) != 0)
        {
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "updateGroupedExpertDownDescriptorTable"))
            return false;

        auto &table = grouped_down_desc_tables_[static_cast<size_t>(descriptor_table_id)];
        if (!table.valid || !table.device_descs ||
            table.num_experts != num_experts ||
            table.d_model != d_model ||
            table.intermediate != intermediate)
        {
            return false;
        }

        uint8_t codebook_id = 0;
        uint32_t codebook_mask = 0;
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &desc = down_descs[expert_id];
            if (isBlankGroupedDesc(desc))
                continue;
            if (!validateCudaGroupedDescShape(desc, d_model, intermediate))
                return false;
            codebook_mask |= cudaGroupedPrefillCodebookBit(desc.codebook_id);
            if (codebook_id == 0)
                codebook_id = desc.codebook_id;
        }
        if (codebook_mask == 0)
            return false;
        if (codebook_mask & (codebook_mask - 1u))
            codebook_id = kCudaMoEMixedCodebookSentinel;
        if (table.codebook_id != codebook_id ||
            table.codebook_mask != codebook_mask)
        {
            LOG_ERROR("[CUDAMoEKernel::updateGroupedExpertDownDescriptorTable] refusing in-place update "
                      "with changed codebook semantics");
            return false;
        }

        const size_t desc_bytes =
            static_cast<size_t>(num_experts) * sizeof(DeviceNativeVNNIMatrixDesc);
        cudaError_t err = cudaMemcpyAsync(table.device_descs, down_descs,
                                          desc_bytes,
                                          cudaMemcpyHostToDevice,
                                          static_cast<cudaStream_t>(getStream()));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel::updateGroupedExpertDownDescriptorTable] descriptor refresh failed: "
                      << cudaGetErrorString(err));
            return false;
        }

        table.host_descs.assign(down_descs, down_descs + num_experts);
        return true;
    }

    bool CUDAMoEKernel::updateGroupedExpertGateUpDescriptorTables(
        int descriptor_table_id,
        const DeviceNativeVNNIMatrixDesc *gate_descs,
        const DeviceNativeVNNIMatrixDesc *up_descs,
        int num_experts,
        int d_model,
        int intermediate)
    {
        if (!gate_descs || !up_descs || descriptor_table_id < 0 ||
            descriptor_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) ||
            num_experts <= 0 || d_model <= 0 || intermediate <= 0 ||
            (d_model % 32) != 0)
        {
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "updateGroupedExpertGateUpDescriptorTables"))
            return false;

        auto &table = grouped_gateup_desc_tables_[static_cast<size_t>(descriptor_table_id)];
        if (!table.valid || !table.device_gate_descs || !table.device_up_descs ||
            table.num_experts != num_experts ||
            table.d_model != d_model ||
            table.intermediate != intermediate)
        {
            return false;
        }

        uint8_t codebook_id = 0;
        uint32_t codebook_mask = 0;
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            const auto &gate_desc = gate_descs[expert_id];
            const auto &up_desc = up_descs[expert_id];
            const bool gate_blank = isBlankGroupedDesc(gate_desc);
            const bool up_blank = isBlankGroupedDesc(up_desc);
            if (gate_blank || up_blank)
            {
                if (gate_blank && up_blank)
                    continue;
                return false;
            }
            if (!gate_desc.valid() || !up_desc.valid() ||
                gate_desc.codebook_id != up_desc.codebook_id ||
                !validateCudaGroupedDescShape(gate_desc, intermediate, d_model) ||
                !validateCudaGroupedDescShape(up_desc, intermediate, d_model))
            {
                return false;
            }
            codebook_mask |= cudaGroupedPrefillCodebookBit(gate_desc.codebook_id);
            if (codebook_id == 0)
                codebook_id = gate_desc.codebook_id;
        }
        if (codebook_mask == 0)
            return false;
        if (codebook_mask & (codebook_mask - 1u))
            codebook_id = kCudaMoEMixedCodebookSentinel;
        if (table.codebook_id != codebook_id ||
            table.codebook_mask != codebook_mask)
        {
            LOG_ERROR("[CUDAMoEKernel::updateGroupedExpertGateUpDescriptorTables] refusing in-place update "
                      "with changed codebook semantics");
            return false;
        }

        const size_t desc_bytes =
            static_cast<size_t>(num_experts) * sizeof(DeviceNativeVNNIMatrixDesc);
        cudaError_t err = cudaMemcpyAsync(table.device_gate_descs, gate_descs,
                                          desc_bytes,
                                          cudaMemcpyHostToDevice,
                                          static_cast<cudaStream_t>(getStream()));
        if (err == cudaSuccess)
            err = cudaMemcpyAsync(table.device_up_descs, up_descs,
                                  desc_bytes,
                                  cudaMemcpyHostToDevice,
                                  static_cast<cudaStream_t>(getStream()));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel::updateGroupedExpertGateUpDescriptorTables] descriptor refresh failed: "
                      << cudaGetErrorString(err));
            return false;
        }

        table.host_gate_descs.assign(gate_descs, gate_descs + num_experts);
        table.host_up_descs.assign(up_descs, up_descs + num_experts);
        return true;
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

        group_active_expert_slots_ = std::min(total_slots, num_experts);
        prepared_num_experts_ = num_experts;
        return true;
    }

    bool CUDAMoEKernel::prepareExpertGroupsAsyncMasked(
        ITensor *routing_indices,
        ITensor *routing_weights,
        int seq_len,
        int num_experts,
        int top_k,
        const uint8_t *expert_mask)
    {
        if (seq_len <= 0 || num_experts <= 0 || top_k <= 0 || !expert_mask)
            return false;
        void *stream = requireStream("CUDAMoEKernel::prepareExpertGroupsAsyncMasked");
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

        uint64_t mask_hash = 1469598103934665603ull;
        for (int i = 0; i < num_experts; ++i)
        {
            mask_hash ^= static_cast<uint64_t>(expert_mask[i]);
            mask_hash *= 1099511628211ull;
        }
        mask_hash ^= static_cast<uint64_t>(num_experts);
        mask_hash *= 1099511628211ull;

        if (!d_group_expert_mask_ || group_expert_mask_cap_ < num_experts)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_expert_mask_),
                                     MoEWorkspaceBuffers::GROUP_EXPERT_MASK,
                                     static_cast<size_t>(num_experts) * sizeof(uint8_t),
                                     "prepareExpertGroupsAsyncMasked(group_expert_mask)"))
            {
                d_group_expert_mask_ = nullptr;
                group_expert_mask_cap_ = 0;
                group_expert_mask_hash_ = 0;
                return false;
            }
            group_expert_mask_cap_ = num_experts;
            group_expert_mask_hash_ = 0;
        }

        if (group_expert_mask_hash_ != mask_hash)
        {
            cudaError_t err = cudaMemcpyAsync(
                d_group_expert_mask_,
                expert_mask,
                static_cast<size_t>(num_experts) * sizeof(uint8_t),
                cudaMemcpyHostToDevice,
                static_cast<cudaStream_t>(stream));
            if (err != cudaSuccess)
            {
                LOG_ERROR("[CUDAMoEKernel::prepareExpertGroupsAsyncMasked] H2D mask copy failed: "
                          << cudaGetErrorString(err));
                return false;
            }
            group_expert_mask_hash_ = mask_hash;
        }

        group_active_expert_slots_ = 0;
        if (!cudaMoE_float_to_masked_int(
                d_float_indices,
                d_group_int_indices_,
                d_group_expert_mask_,
                total_slots,
                num_experts,
                device_ordinal_,
                stream))
        {
            return false;
        }

        if (!groupTokensByExpertDevice(d_group_int_indices_, d_float_weights,
                                       seq_len, num_experts, top_k,
                                       d_group_offsets_, d_group_counts_,
                                       d_group_token_indices_, d_group_weights_))
            return false;

        group_active_expert_slots_ = std::min(total_slots, num_experts);
        prepared_num_experts_ = num_experts;
        return true;
    }

    bool CUDAMoEKernel::updateGroupedPrefillExpertMask(
        const uint8_t *expert_mask,
        int num_experts)
    {
        if (num_experts <= 0 || !expert_mask)
            return false;

        void *stream = requireStream("CUDAMoEKernel::updateGroupedPrefillExpertMask");
        if (!stream)
            return false;

        uint64_t mask_hash = 1469598103934665603ull;
        for (int i = 0; i < num_experts; ++i)
        {
            mask_hash ^= static_cast<uint64_t>(expert_mask[i]);
            mask_hash *= 1099511628211ull;
        }
        mask_hash ^= static_cast<uint64_t>(num_experts);
        mask_hash *= 1099511628211ull;

        if (!d_group_expert_mask_ || group_expert_mask_cap_ < num_experts)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_group_expert_mask_),
                                     MoEWorkspaceBuffers::GROUP_EXPERT_MASK,
                                     static_cast<size_t>(num_experts) * sizeof(uint8_t),
                                     "updateGroupedPrefillExpertMask(group_expert_mask)"))
            {
                d_group_expert_mask_ = nullptr;
                group_expert_mask_cap_ = 0;
                group_expert_mask_hash_ = 0;
                return false;
            }
            group_expert_mask_cap_ = num_experts;
            group_expert_mask_hash_ = 0;
        }

        if (group_expert_mask_hash_ == mask_hash)
            return true;

        cudaError_t err = cudaMemcpyAsync(
            d_group_expert_mask_,
            expert_mask,
            static_cast<size_t>(num_experts) * sizeof(uint8_t),
            cudaMemcpyHostToDevice,
            static_cast<cudaStream_t>(stream));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel::updateGroupedPrefillExpertMask] H2D mask copy failed: "
                      << cudaGetErrorString(err));
            return false;
        }

        group_expert_mask_hash_ = mask_hash;
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
                d_group_original_to_grouped_,
                d_group_original_expert_ids_,
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
        const int active_expert_slots = group_active_expert_slots_;
        const int *d_active_expert_ids =
            (active_expert_slots > 0) ? d_group_active_expert_ids_ : nullptr;
        const bool use_gateup_kpart =
            active_expert_slots > 0 &&
            max_tokens_per_expert <= 4 &&
            debugEnv().gemm.cuda_moe_gateup_kpart_decode;
        if (use_gateup_kpart &&
            !ensureGroupedGateUpKPartScratchCapacity(
                total_slots,
                debugEnv().gemm.cuda_moe_gateup_kparts,
                intermediate))
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillPipeline] "
                      "verifier grouped gate/up split-K scratch allocation failed");
            return false;
        }
        const bool use_down_ordered_kpart =
            active_expert_slots > 0 &&
            max_tokens_per_expert <= 4 &&
            d_group_original_to_grouped_ != nullptr &&
            d_group_original_expert_ids_ != nullptr &&
            debugEnv().gemm.cuda_moe_down_kpart_decode;
        if (use_down_ordered_kpart &&
            !ensureGroupedDownKPartScratchCapacity(
                debugEnv().gemm.cuda_moe_down_kparts,
                d_model,
                seq_len))
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillPipeline] "
                      "verifier grouped down split-K scratch allocation failed");
            return false;
        }
        if (!ensureGroupedPrefillScratchCapacity(total_slots, d_model, intermediate) ||
            !ensureTensorOnDevice(hidden, device, stream, "hidden") ||
            !ensureOutputOnDevice(output, device, stream, "output"))
            return false;

        const float *d_hidden = static_cast<const float *>(hidden->gpu_data_ptr());
        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!d_hidden || !d_output)
            return false;

        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        /*
         * Ordered scatter overwrites every output element, including the shared
         * expert identity-map case.  Only the atomic scatter fallback needs a
         * zeroed destination before accumulation.  CUDA currently uses ordered
         * scatter for this graph-native prefill path whenever the grouping map
         * is present, so the explicit alias keeps the pre-zero contract aligned
         * with ROCm and protects future atomic variants from silently skipping
         * the clear.
         */
        const bool ordered_scatter_overwrites_output =
            active_expert_slots > 0 && d_group_original_to_grouped_ != nullptr;
        const bool scatter_overwrites_output = ordered_scatter_overwrites_output;
        if (!scatter_overwrites_output)
        {
            cudaError_t err = cudaMemsetAsync(d_output, 0,
                                              static_cast<size_t>(seq_len) * d_model * sizeof(float),
                                              cuda_stream);
            if (err != cudaSuccess)
            {
                LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillPipeline] output memset failed: "
                          << cudaGetErrorString(err));
                return false;
            }
        }

        const bool ok = cudaMoE_grouped_prefill_pipeline(
            d_hidden,
            gateup_table.device_gate_descs,
            gateup_table.device_up_descs,
            down_table.device_descs,
            d_group_counts_,
            d_group_offsets_,
            d_group_token_indices_,
            ordered_scatter_overwrites_output ? d_group_original_to_grouped_ : nullptr,
            use_down_ordered_kpart ? d_group_original_expert_ids_ : nullptr,
            d_active_expert_ids,
            d_group_weights_,
            d_prefill_A_int8_,
            d_prefill_A_scales_,
            d_prefill_gate_,
            d_prefill_up_,
            use_gateup_kpart ? d_grouped_gateup_gate_partials_ : nullptr,
            use_gateup_kpart ? d_grouped_gateup_up_partials_ : nullptr,
            d_prefill_swiglu_int8_,
            d_prefill_swiglu_scales_,
            use_down_ordered_kpart ? d_grouped_down_partials_ : nullptr,
            d_prefill_gate_,
            d_output,
            num_experts,
            d_model,
            intermediate,
            max_tokens_per_expert,
            total_slots,
            top_k,
            active_expert_slots,
            0,
            gateup_table.codebook_id,
            down_table.codebook_id,
            gateup_table.codebook_mask,
            down_table.codebook_mask,
            use_gateup_kpart ? debugEnv().gemm.cuda_moe_gateup_kparts : 0,
            use_down_ordered_kpart ? debugEnv().gemm.cuda_moe_down_kparts : 0,
            device_ordinal_,
            stream);
        if (!ok)
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillPipeline] grouped CUDA pipeline failed");
            return false;
        }

        markDeviceWritten(output, device, stream);
        const int selected_tile_m = selectGroupedPrefillTileM(
            debugEnv().gemm.cuda_moe_prefill_tile_m, max_tokens_per_expert);
        const int selected_tile_n =
            (active_expert_slots > 0 && max_tokens_per_expert <= 4) ? 64 : 128;
        recordGroupedPrefillCounters(
            seq_len,
            top_k,
            num_experts,
            active_expert_slots,
            selected_tile_m,
            selected_tile_n,
            use_gateup_kpart,
            debugEnv().gemm.cuda_moe_prefill_fuse_swiglu,
            use_down_ordered_kpart,
            ordered_scatter_overwrites_output);
        return true;
    }

    bool CUDAMoEKernel::executeGroupedPrefillPipelineFromRuntime(
        DeviceMoELayerRuntime *device_runtime_layer,
        const DeviceMoELayerRuntime &runtime_host_layer,
        ITensor *hidden, ITensor *output,
        int gateup_desc_table_id,
        int down_desc_table_id,
        int seq_len, int d_model, int intermediate,
        int num_experts, int top_k)
    {
        if (!device_runtime_layer)
            return false;
        if (seq_len <= 0 || d_model <= 0 || intermediate <= 0 || num_experts <= 0 || top_k <= 0)
            return false;
        if (runtime_host_layer.expert_count != static_cast<uint32_t>(num_experts) ||
            runtime_host_layer.top_k != static_cast<uint32_t>(top_k) ||
            runtime_host_layer.prefill_token_capacity < static_cast<uint32_t>(seq_len) ||
            runtime_host_layer.prefill_route_capacity < static_cast<uint32_t>(seq_len * top_k) ||
            !runtime_host_layer.expert_counts ||
            !runtime_host_layer.expert_offsets ||
            !runtime_host_layer.grouped_token_ids ||
            !runtime_host_layer.grouped_route_weights)
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillPipelineFromRuntime] invalid runtime scratch contract"
                      << " expert_count=" << runtime_host_layer.expert_count
                      << " expected_experts=" << num_experts
                      << " top_k=" << runtime_host_layer.top_k
                      << " expected_top_k=" << top_k
                      << " token_capacity=" << runtime_host_layer.prefill_token_capacity
                      << " seq_len=" << seq_len
                      << " route_capacity=" << runtime_host_layer.prefill_route_capacity
                      << " total_slots=" << (seq_len * top_k));
            return false;
        }
        if (gateup_desc_table_id < 0 ||
            gateup_desc_table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()) ||
            down_desc_table_id < 0 ||
            down_desc_table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillPipelineFromRuntime] invalid descriptor table id");
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
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillPipelineFromRuntime] descriptor table shape mismatch");
            return false;
        }

        void *stream = requireStream("CUDAMoEKernel::executeGroupedPrefillPipelineFromRuntime");
        const DeviceId device = deviceId();
        const int total_slots = seq_len * top_k;
        const int max_tokens_per_expert = seq_len;
        const int active_expert_slots = std::min(total_slots, num_experts);
        const bool use_gateup_kpart =
            debugEnv().gemm.cuda_moe_gateup_kpart_decode &&
            active_expert_slots > 0 &&
            max_tokens_per_expert <= 4;
        /*
         * Runtime grouped verifier prefill must mirror the public M=1 runtime
         * decode route.  CUDA serial decode uses split-K gate/up for tiny MoE
         * rows by default, so M=2..4 publication has to use the same split-K
         * partial ordering before it evaluates SwiGLU and quantizes the down
         * input.  Keeping this as a grouped kernel sequence preserves the target
         * architecture without reintroducing row replay.
         */
        if (use_gateup_kpart &&
            !ensureGroupedGateUpKPartScratchCapacity(
                total_slots,
                debugEnv().gemm.cuda_moe_gateup_kparts,
                intermediate))
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillPipelineFromRuntime] "
                      "verifier grouped gate/up split-K scratch allocation failed");
            return false;
        }
        const bool use_down_ordered_kpart =
            debugEnv().gemm.cuda_moe_down_kpart_decode &&
            max_tokens_per_expert <= 4 &&
            num_experts > 1 &&
            top_k > 1;
        if (use_down_ordered_kpart &&
            !runtime_host_layer.route_expert_ids)
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillPipelineFromRuntime] "
                      "verifier grouped down split-K requires runtime route expert ids");
            return false;
        }
        if (use_down_ordered_kpart &&
            !ensureGroupedDownKPartScratchCapacity(
                debugEnv().gemm.cuda_moe_down_kparts,
                d_model,
                seq_len))
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillPipelineFromRuntime] "
                      "verifier grouped down split-K scratch allocation failed");
            return false;
        }
        if (!ensureGroupingBufferCapacity(total_slots, num_experts) ||
            !ensureGroupedPrefillScratchCapacity(total_slots, d_model, intermediate) ||
            !ensureRuntimePrefillDescriptorCapacity(num_experts) ||
            !ensureTensorOnDevice(hidden, device, stream, "hidden") ||
            !ensureOutputOnDevice(output, device, stream, "output"))
        {
            return false;
        }

        if (!cudaMoE_materialize_runtime_prefill_descriptor_tables(
                device_runtime_layer,
                d_runtime_prefill_gate_descs_,
                d_runtime_prefill_up_descs_,
                d_runtime_prefill_down_descs_,
                num_experts,
                device_ordinal_,
                stream))
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillPipelineFromRuntime] failed to materialize runtime descriptors");
            return false;
        }

        const float *d_hidden = static_cast<const float *>(hidden->gpu_data_ptr());
        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!d_hidden || !d_output)
            return false;

        if (!cudaMoE_build_active_expert_list_runtime(
                device_runtime_layer,
                d_group_active_expert_ids_,
                num_experts,
                active_expert_slots,
                device_ordinal_,
                stream))
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillPipelineFromRuntime] failed to build active expert list");
            return false;
        }
        group_active_expert_slots_ = active_expert_slots;

        /*
         * Runtime grouped prefill stores route slots in grouped_token_ids because
         * LLEP first balances individual top-k router choices and then regroups
         * only choices local to this participant.  Rebuilding the ordered scatter
         * map from the runtime scratch gives LLEP the same deterministic top-k
         * accumulation order as ordinary grouped prefill and avoids prefix-cache
         * drift from atomicAdd ordering.
         */
        if (!cudaMoE_build_runtime_original_to_grouped(
                device_runtime_layer,
                d_group_original_to_grouped_,
                total_slots,
                total_slots,
                num_experts,
                top_k,
                device_ordinal_,
                stream))
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillPipelineFromRuntime] failed to build runtime ordered-scatter map");
            return false;
        }

        const bool ok = cudaMoE_grouped_prefill_pipeline(
            d_hidden,
            d_runtime_prefill_gate_descs_,
            d_runtime_prefill_up_descs_,
            d_runtime_prefill_down_descs_,
            runtime_host_layer.expert_counts,
            runtime_host_layer.expert_offsets,
            runtime_host_layer.grouped_token_ids,
            d_group_original_to_grouped_,
            use_down_ordered_kpart ? runtime_host_layer.route_expert_ids : nullptr,
            d_group_active_expert_ids_,
            runtime_host_layer.grouped_route_weights,
            d_prefill_A_int8_,
            d_prefill_A_scales_,
            d_prefill_gate_,
            d_prefill_up_,
            use_gateup_kpart ? d_grouped_gateup_gate_partials_ : nullptr,
            use_gateup_kpart ? d_grouped_gateup_up_partials_ : nullptr,
            d_prefill_swiglu_int8_,
            d_prefill_swiglu_scales_,
            use_down_ordered_kpart ? d_grouped_down_partials_ : nullptr,
            d_prefill_gate_,
            d_output,
            num_experts,
            d_model,
            intermediate,
            max_tokens_per_expert,
            total_slots,
            top_k,
            active_expert_slots,
            1,
            gateup_table.codebook_id,
            down_table.codebook_id,
            gateup_table.codebook_mask,
            down_table.codebook_mask,
            use_gateup_kpart ? debugEnv().gemm.cuda_moe_gateup_kparts : 0,
            use_down_ordered_kpart ? debugEnv().gemm.cuda_moe_down_kparts : 0,
            device_ordinal_,
            stream);
        if (!ok)
        {
            LOG_ERROR("[CUDAMoEKernel::executeGroupedPrefillPipelineFromRuntime] grouped CUDA pipeline failed");
            return false;
        }

        markDeviceWritten(output, device, stream);
        recordGroupedPrefillCounters(
            seq_len,
            top_k,
            num_experts,
            active_expert_slots,
            selectGroupedPrefillTileM(debugEnv().gemm.cuda_moe_prefill_tile_m, max_tokens_per_expert),
            (active_expert_slots > 0 && max_tokens_per_expert <= 4) ? 64 : 128,
            use_gateup_kpart,
            debugEnv().gemm.cuda_moe_prefill_fuse_swiglu,
            use_down_ordered_kpart,
            true);
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
        const bool capture_active = isCudaMoEDecodeCaptureActive(stream);
        const DeviceId device = deviceId();
        if (!setMoEDevice(device_ordinal_, "groupedExpertGateUpDecodeFromTable"))
            return false;
        if (!requireTensorType(input, TensorType::FP32, "input", "groupedExpertGateUpDecodeFromTable") ||
            !requireMatrixCapacity(input, 1, d_model, "input", "groupedExpertGateUpDecodeFromTable") ||
            !requireTensorElements(input, static_cast<size_t>(d_model), "input", "groupedExpertGateUpDecodeFromTable"))
            return false;

        const int k_partitions = debugEnv().gemm.cuda_moe_gateup_kparts;
        const bool use_kpart = debugEnv().gemm.cuda_moe_gateup_kpart_decode;
        if (use_kpart && !ensureGroupedGateUpKPartScratchCapacity(num_active, k_partitions, intermediate))
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromTable] "
                      "K-part gate/up decode was requested but scratch allocation failed");
            return false;
        }
        if (!ensureGroupedGateUpDecodeCapacity(num_active, d_model))
            return false;

        const float *d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        if (!d_hidden && capture_active)
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
                capture_active)
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
        if (!ensureRuntimeGateUpPointerArrays(table_id, RuntimePointerArrayScope::TableDecode,
                                              num_active, gate_ptrs, up_ptrs,
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
                                  false,
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
                                  false,
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
        const bool capture_active = isCudaMoEDecodeCaptureActive(stream);
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

        if (!output->gpu_data_ptr() && capture_active)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromTable] output allocation required during graph capture");
            return false;
        }
        if (!ensureOutputOnDevice(output, device, stream, "moe_output"))
            return false;

        const float **d_gate_ptrs = nullptr;
        const float **d_up_ptrs = nullptr;
        if (!ensureRuntimeDownPointerArrays(table_id, RuntimePointerArrayScope::TableDecode,
                                            num_active, gate_ptrs, up_ptrs,
                                            &d_gate_ptrs, &d_up_ptrs))
            return false;

        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        const int k_partitions = debugEnv().gemm.cuda_moe_down_kparts;
        const bool use_kpart = debugEnv().gemm.cuda_moe_down_kpart_decode;
        if (use_kpart && !ensureGroupedDownKPartScratchCapacity(k_partitions, d_model, num_active))
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromTable] "
                      "K-part down decode was requested but scratch allocation failed");
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

    bool CUDAMoEKernel::groupedExpertGateUpDecodeFromRouting(
        const TensorBase *input,
        ITensor *routing_indices,
        int table_id,
        int top_k,
        ITensor *const *gate_outputs,
        ITensor *const *up_outputs,
        int d_model,
        int intermediate,
        const uint8_t *expert_mask)
    {
        if (!input || !routing_indices || table_id < 0 || top_k <= 0 ||
            !gate_outputs || !up_outputs || d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (top_k > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            (d_model % 32) != 0)
        {
            return false;
        }
        if (table_id >= static_cast<int>(grouped_gateup_desc_tables_.size()))
            return false;

        const auto &table = grouped_gateup_desc_tables_[table_id];
        if (!table.valid || !table.device_gate_descs || !table.device_up_descs ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRouting] descriptor table mismatch");
            return false;
        }

        void *stream = requireStream("CUDAMoEKernel::groupedExpertGateUpDecodeFromRouting");
        const bool capture_active = isCudaMoEDecodeCaptureActive(stream);
        const DeviceId device = deviceId();
        if (!setMoEDevice(device_ordinal_, "groupedExpertGateUpDecodeFromRouting"))
            return false;
        if (!ensureRoutingDecodeMetadataCapacity(top_k) ||
            !ensureGroupedGateUpDecodeCapacity(top_k, d_model))
        {
            return false;
        }

        const int k_partitions = debugEnv().gemm.cuda_moe_gateup_kparts;
        const bool use_kpart = debugEnv().gemm.cuda_moe_gateup_kpart_decode;
        if (use_kpart && !ensureGroupedGateUpKPartScratchCapacity(top_k, k_partitions, intermediate))
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRouting] "
                      "K-part gate/up decode was requested but scratch allocation failed");
            return false;
        }

        const float *d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        if (!d_hidden && capture_active)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRouting] input upload required during graph capture");
            return false;
        }
        if (!d_hidden)
        {
            auto *mutable_input = const_cast<TensorBase *>(input);
            if (!mutable_input->ensureOnDevice(device, stream))
                return false;
            d_hidden = static_cast<const float *>(mutable_input->gpu_data_ptr());
        }

        if (!routing_indices->gpu_data_ptr() && capture_active)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRouting] routing upload required during graph capture");
            return false;
        }
        if (!ensureTensorOnDevice(routing_indices, device, stream, "routing_indices"))
            return false;
        const float *d_routing_indices =
            static_cast<const float *>(routing_indices->gpu_data_ptr());
        if (!d_hidden || !d_routing_indices)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRouting] missing input/routing device pointer");
            return false;
        }

        /*
         * Dynamic/LLEP verifier rows still consume the device-owned routing
         * tensor, but each LocalTP participant must contribute only the experts
         * it owns.  The masked conversion preserves the original routing tensor
         * for histograms/runtime publication and marks nonlocal top-k slots as
         * inactive in backend-owned scratch.  The grouped kernels already treat
         * expert id -1 as a no-op slot.
         */
        if (expert_mask)
        {
            if (!updateGroupedPrefillExpertMask(expert_mask, table.num_experts))
                return false;
            if (!cudaMoE_float_to_masked_int(
                    d_routing_indices,
                    d_routing_decode_expert_ids_,
                    d_group_expert_mask_,
                    top_k,
                    table.num_experts,
                    device_ordinal_,
                    stream))
            {
                return false;
            }
        }
        else if (!cudaMoE_float_to_int(
                     d_routing_indices,
                     d_routing_decode_expert_ids_,
                     top_k,
                     device_ordinal_,
                     stream))
        {
            return false;
        }

        std::array<float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        for (int slot = 0; slot < top_k; ++slot)
        {
            if (!gate_outputs[slot] || !up_outputs[slot])
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRouting] null output tensor for slot "
                          << slot);
                return false;
            }
            if ((!gate_outputs[slot]->gpu_data_ptr() || !up_outputs[slot]->gpu_data_ptr()) &&
                capture_active)
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRouting] output allocation required during graph capture");
                return false;
            }
            if (!ensureOutputOnDevice(gate_outputs[slot], device, stream, "gate_output") ||
                !ensureOutputOnDevice(up_outputs[slot], device, stream, "up_output"))
            {
                return false;
            }

            gate_ptrs[slot] = static_cast<float *>(gate_outputs[slot]->gpu_data_ptr());
            up_ptrs[slot] = static_cast<float *>(up_outputs[slot]->gpu_data_ptr());
            if (!gate_ptrs[slot] || !up_ptrs[slot])
                return false;
        }

        float **d_gate_ptrs = nullptr;
        float **d_up_ptrs = nullptr;
        if (!ensureRuntimeGateUpPointerArrays(
                table_id, RuntimePointerArrayScope::TableDecode,
                top_k, gate_ptrs, up_ptrs,
                &d_gate_ptrs, &d_up_ptrs))
        {
            return false;
        }

        const bool ok = use_kpart
                            ? cudaMoE_grouped_gate_up_native_vnni_decode_table_kpart(
                                  d_hidden,
                                  table.device_gate_descs,
                                  table.device_up_descs,
                                  d_routing_decode_expert_ids_,
                                  d_gate_ptrs,
                                  d_up_ptrs,
                                  d_decode_hidden_int8_,
                                  d_decode_hidden_scales_,
                                  false,
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
                                  d_routing_decode_expert_ids_,
                                  d_gate_ptrs,
                                  d_up_ptrs,
                                  d_decode_hidden_int8_,
                                  d_decode_hidden_scales_,
                                  false,
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
                "cuda_moe_grouped_decode_gateup_calls", "routing", top_k,
                d_model, intermediate, use_kpart ? "kpart" : "serial");
        }
        return ok;
    }

    bool CUDAMoEKernel::groupedExpertDownDecodeFromRouting(
        ITensor *const *gate_tensors,
        ITensor *const *up_tensors,
        ITensor *routing_indices,
        ITensor *routing_weights,
        int table_id,
        int top_k,
        ITensor *output,
        int d_model,
        int intermediate,
        const uint8_t *expert_mask)
    {
        if (!gate_tensors || !up_tensors || !routing_indices || !routing_weights ||
            table_id < 0 || top_k <= 0 || !output ||
            d_model <= 0 || intermediate <= 0)
        {
            return false;
        }
        if (top_k > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            (intermediate % 32) != 0)
        {
            return false;
        }
        if (table_id >= static_cast<int>(grouped_down_desc_tables_.size()))
            return false;

        const auto &table = grouped_down_desc_tables_[table_id];
        if (!table.valid || !table.device_descs ||
            table.d_model != d_model || table.intermediate != intermediate)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRouting] descriptor table mismatch");
            return false;
        }

        void *stream = requireStream("CUDAMoEKernel::groupedExpertDownDecodeFromRouting");
        const bool capture_active = isCudaMoEDecodeCaptureActive(stream);
        const DeviceId device = deviceId();
        if (!setMoEDevice(device_ordinal_, "groupedExpertDownDecodeFromRouting"))
            return false;
        if (!ensureRoutingDecodeMetadataCapacity(top_k) ||
            !ensureGroupedDownDecodeCapacity(top_k, intermediate))
        {
            return false;
        }

        if ((!routing_indices->gpu_data_ptr() || !routing_weights->gpu_data_ptr()) &&
            capture_active)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRouting] routing upload required during graph capture");
            return false;
        }
        if (!ensureTensorOnDevice(routing_indices, device, stream, "routing_indices") ||
            !ensureTensorOnDevice(routing_weights, device, stream, "routing_weights"))
        {
            return false;
        }

        const float *d_routing_indices =
            static_cast<const float *>(routing_indices->gpu_data_ptr());
        const float *d_weights =
            static_cast<const float *>(routing_weights->gpu_data_ptr());
        if (!d_routing_indices || !d_weights)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRouting] missing routing device pointer");
            return false;
        }

        if (expert_mask)
        {
            if (!updateGroupedPrefillExpertMask(expert_mask, table.num_experts))
                return false;
            if (!cudaMoE_float_to_masked_int(
                    d_routing_indices,
                    d_routing_decode_expert_ids_,
                    d_group_expert_mask_,
                    top_k,
                    table.num_experts,
                    device_ordinal_,
                    stream))
            {
                return false;
            }
        }
        else if (!cudaMoE_float_to_int(
                     d_routing_indices,
                     d_routing_decode_expert_ids_,
                     top_k,
                     device_ordinal_,
                     stream))
        {
            return false;
        }

        std::array<const float *, kRuntimePointerArrayMaxTopK> gate_ptrs = {};
        std::array<const float *, kRuntimePointerArrayMaxTopK> up_ptrs = {};
        for (int slot = 0; slot < top_k; ++slot)
        {
            if (!gate_tensors[slot] || !up_tensors[slot])
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRouting] null gate/up tensor for slot "
                          << slot);
                return false;
            }
            gate_ptrs[slot] = static_cast<const float *>(gate_tensors[slot]->gpu_data_ptr());
            up_ptrs[slot] = static_cast<const float *>(up_tensors[slot]->gpu_data_ptr());
            if (!gate_ptrs[slot] || !up_ptrs[slot])
            {
                LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRouting] missing gate/up device pointer for slot "
                          << slot);
                return false;
            }
        }

        if (!output->gpu_data_ptr() && capture_active)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRouting] output allocation required during graph capture");
            return false;
        }
        if (!ensureOutputOnDevice(output, device, stream, "moe_output"))
            return false;

        const float **d_gate_ptrs = nullptr;
        const float **d_up_ptrs = nullptr;
        if (!ensureRuntimeDownPointerArrays(
                table_id, RuntimePointerArrayScope::TableDecode,
                top_k, gate_ptrs, up_ptrs,
                &d_gate_ptrs, &d_up_ptrs))
        {
            return false;
        }

        float *d_output = static_cast<float *>(output->gpu_data_ptr());
        if (!d_output)
            return false;

        const int k_partitions = debugEnv().gemm.cuda_moe_down_kparts;
        const bool use_kpart = debugEnv().gemm.cuda_moe_down_kpart_decode;
        if (use_kpart && !ensureGroupedDownKPartScratchCapacity(k_partitions, d_model, top_k))
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRouting] "
                      "K-part down decode was requested but scratch allocation failed");
            return false;
        }

        const bool ok = use_kpart
            ? cudaMoE_grouped_swiglu_down_native_vnni_decode_table_kpart(
                  d_gate_ptrs,
                  d_up_ptrs,
                  table.device_descs,
                  d_routing_decode_expert_ids_,
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
                  d_routing_decode_expert_ids_,
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
                "cuda_moe_grouped_decode_down_calls", "routing", top_k,
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
        int intermediate,
        MoEDecodeDescriptorSource descriptor_source)
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
        const bool capture_active = isCudaMoEDecodeCaptureActive(stream);
        const DeviceId device = deviceId();
        if (!setMoEDevice(device_ordinal_, "groupedExpertDecodeFromRuntime"))
            return false;

        const int gateup_k_partitions = debugEnv().gemm.cuda_moe_gateup_kparts;
        const bool use_gateup_kpart = debugEnv().gemm.cuda_moe_gateup_kpart_decode;
        if (use_gateup_kpart &&
            !ensureGroupedGateUpKPartScratchCapacity(top_k, gateup_k_partitions, intermediate))
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDecodeFromRuntime] "
                      "K-part gate/up decode was requested but scratch allocation failed");
            return false;
        }
        const int down_k_partitions = debugEnv().gemm.cuda_moe_down_kparts;
        const bool use_down_kpart = debugEnv().gemm.cuda_moe_down_kpart_decode;
        if (use_down_kpart &&
            !ensureGroupedDownKPartScratchCapacity(down_k_partitions, d_model, top_k))
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDecodeFromRuntime] "
                      "K-part down decode was requested but scratch allocation failed");
            return false;
        }

        if (!ensureGroupedPrefillScratchCapacity(top_k, d_model, intermediate) ||
            !ensureGroupedGateUpDecodeCapacity(top_k, d_model) ||
            !ensureGroupedDownDecodeCapacity(top_k, intermediate))
            return false;

        const float *d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        if (!d_hidden && capture_active)
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

        if (!output->gpu_data_ptr() && capture_active)
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
        if (!ensureRuntimeGateUpPointerArrays(gateup_table_id, RuntimePointerArrayScope::RuntimeFused,
                                              top_k, gate_ptrs, up_ptrs,
                                              &d_gate_ptrs, &d_up_ptrs))
            return false;
        const float **d_down_gate_ptrs = nullptr;
        const float **d_down_up_ptrs = nullptr;
        if (!ensureRuntimeDownPointerArrays(down_table_id, RuntimePointerArrayScope::RuntimeFused,
                                            top_k, const_gate_ptrs, const_up_ptrs,
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

        const bool use_runtime_descriptors =
            descriptor_source == MoEDecodeDescriptorSource::RuntimePlacementTable;
        const bool reuse_router_q8_hidden =
            debugEnv().gemm.cuda_moe_reuse_router_q8_hidden &&
            !capture_active &&
            router_q8_hidden_valid_ &&
            router_q8_hidden_source_ == d_hidden &&
            d_decode_hidden_int8_ &&
            d_decode_hidden_scales_ &&
            decode_gateup_d_model_cap_ >= d_model;
        if (reuse_router_q8_hidden)
        {
            PerfStatsCollector::addCounter(
                "kernel", "cuda_moe_gateup_reused_router_q8_hidden_calls", 1.0, {}, {},
                {{"top_k", std::to_string(top_k)},
                 {"d_model", std::to_string(d_model)}});
        }
        const bool gateup_ok = use_runtime_descriptors
                                   ? (use_gateup_kpart
                                          ? cudaMoE_grouped_gate_up_native_vnni_decode_runtime_kpart(
                                                d_hidden,
                                                runtime_layer,
                                                d_expert_ids,
                                                d_gate_ptrs,
                                                d_up_ptrs,
                                                d_decode_hidden_int8_,
                                                d_decode_hidden_scales_,
                                                reuse_router_q8_hidden,
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
                                          : cudaMoE_grouped_gate_up_native_vnni_decode_runtime(
                                                d_hidden,
                                                runtime_layer,
                                                d_expert_ids,
                                                d_gate_ptrs,
                                                d_up_ptrs,
                                                d_decode_hidden_int8_,
                                                d_decode_hidden_scales_,
                                                reuse_router_q8_hidden,
                                                top_k,
                                                intermediate,
                                                d_model,
                                                gateup_table.num_experts,
                                                gateup_table.codebook_id,
                                                device_ordinal_,
                                                stream))
                                   : (use_gateup_kpart
                                          ? cudaMoE_grouped_gate_up_native_vnni_decode_table_kpart(
                                                d_hidden,
                                                gateup_table.device_gate_descs,
                                                gateup_table.device_up_descs,
                                                d_expert_ids,
                                                d_gate_ptrs,
                                                d_up_ptrs,
                                                d_decode_hidden_int8_,
                                                d_decode_hidden_scales_,
                                                reuse_router_q8_hidden,
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
                                                reuse_router_q8_hidden,
                                                top_k,
                                                intermediate,
                                                d_model,
                                                gateup_table.codebook_id,
                                                device_ordinal_,
                                                stream));
        if (!gateup_ok)
            return false;

        const bool down_ok = use_runtime_descriptors
                                 ? (use_down_kpart
                                        ? cudaMoE_grouped_swiglu_down_native_vnni_decode_runtime_kpart(
                                              d_down_gate_ptrs,
                                              d_down_up_ptrs,
                                              runtime_layer,
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
                                        : cudaMoE_grouped_swiglu_down_native_vnni_decode_runtime(
                                              d_down_gate_ptrs,
                                              d_down_up_ptrs,
                                              runtime_layer,
                                              d_expert_ids,
                                              d_weights,
                                              d_decode_swiglu_int8_,
                                              d_decode_swiglu_scales_,
                                              d_output,
                                              top_k,
                                              d_model,
                                              intermediate,
                                              down_table.num_experts,
                                              down_table.codebook_id,
                                              device_ordinal_,
                                              stream))
                                 : (use_down_kpart
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
                                              stream));
        if (!down_ok)
            return false;

        markDeviceWritten(output, device, stream);
        recordGroupedDecodeCounter(
            "cuda_moe_grouped_decode_fused_calls",
            use_runtime_descriptors ? "runtime" : "runtime_static_table",
            top_k, d_model, intermediate, "fused_block_down");
        if (!capture_active)
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
        const bool capture_active = isCudaMoEDecodeCaptureActive(stream);
        const DeviceId device = deviceId();
        if (!ensureGroupedGateUpDecodeCapacity(top_k, d_model))
            return false;

        // K-part decode is an explicit contract. If it is enabled and the
        // partial scratch cannot be sized, fail instead of changing routes.
        const int k_partitions = debugEnv().gemm.cuda_moe_gateup_kparts;
        const bool use_kpart = debugEnv().gemm.cuda_moe_gateup_kpart_decode;
        if (use_kpart && !ensureGroupedGateUpKPartScratchCapacity(top_k, k_partitions, intermediate))
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRuntime] "
                      "K-part gate/up decode was requested but scratch allocation failed");
            return false;
        }

        if (!setMoEDevice(device_ordinal_, "groupedExpertGateUpDecodeFromRuntime"))
            return false;

        const float *d_hidden = static_cast<const float *>(input->gpu_data_ptr());
        if (!d_hidden)
        {
            if (capture_active)
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
                capture_active)
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
        if (!ensureRuntimeGateUpPointerArrays(table_id, RuntimePointerArrayScope::RuntimeTwoStep,
                                              top_k, gate_ptrs, up_ptrs,
                                              &d_gate_ptrs, &d_up_ptrs))
            return false;

        const int *d_expert_ids = runtimeTopKExpertIdsDevice(runtime_layer);
        if (!d_hidden || !d_expert_ids)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertGateUpDecodeFromRuntime] missing runtime/input device pointer");
            return false;
        }

        const bool reuse_router_q8_hidden =
            debugEnv().gemm.cuda_moe_reuse_router_q8_hidden &&
            !capture_active &&
            router_q8_hidden_valid_ &&
            router_q8_hidden_source_ == d_hidden &&
            d_decode_hidden_int8_ &&
            d_decode_hidden_scales_ &&
            decode_gateup_d_model_cap_ >= d_model;
        if (reuse_router_q8_hidden)
        {
            PerfStatsCollector::addCounter(
                "kernel", "cuda_moe_gateup_reused_router_q8_hidden_calls", 1.0, {}, {},
                {{"top_k", std::to_string(top_k)},
                 {"d_model", std::to_string(d_model)}});
        }

        const bool ok = use_kpart
                            ? cudaMoE_grouped_gate_up_native_vnni_decode_runtime_kpart(
                                  d_hidden,
                                  runtime_layer,
                                  d_expert_ids,
                                  d_gate_ptrs,
                                  d_up_ptrs,
                                  d_decode_hidden_int8_,
                                  d_decode_hidden_scales_,
                                  reuse_router_q8_hidden,
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
                            : cudaMoE_grouped_gate_up_native_vnni_decode_runtime(
                                  d_hidden,
                                  runtime_layer,
                                  d_expert_ids,
                                  d_gate_ptrs,
                                  d_up_ptrs,
                                  d_decode_hidden_int8_,
                                  d_decode_hidden_scales_,
                                  reuse_router_q8_hidden,
                                  top_k,
                                  intermediate,
                                  d_model,
                                  table.num_experts,
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
        const bool capture_active = isCudaMoEDecodeCaptureActive(stream);
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

        if (!output->gpu_data_ptr() && capture_active)
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRuntime] output allocation required during graph capture");
            return false;
        }
        if (!ensureOutputOnDevice(output, device, stream, "moe_output"))
            return false;

        const float **d_gate_ptrs = nullptr;
        const float **d_up_ptrs = nullptr;
        if (!ensureRuntimeDownPointerArrays(table_id, RuntimePointerArrayScope::RuntimeTwoStep,
                                            top_k, gate_ptrs, up_ptrs,
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
        // count by k_partitions. Serial is only used when the K-part toggle is
        // explicitly off.
        const int k_partitions = debugEnv().gemm.cuda_moe_down_kparts;
        const bool use_kpart = debugEnv().gemm.cuda_moe_down_kpart_decode;
        if (use_kpart && !ensureGroupedDownKPartScratchCapacity(k_partitions, d_model, top_k))
        {
            LOG_ERROR("[CUDAMoEKernel::groupedExpertDownDecodeFromRuntime] "
                      "K-part down decode was requested but scratch allocation failed");
            return false;
        }

        const bool ok = use_kpart
            ? cudaMoE_grouped_swiglu_down_native_vnni_decode_runtime_kpart(
                  d_gate_ptrs,
                  d_up_ptrs,
                  runtime_layer,
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
            : cudaMoE_grouped_swiglu_down_native_vnni_decode_runtime(
                  d_gate_ptrs,
                  d_up_ptrs,
                  runtime_layer,
                  d_expert_ids,
                  d_weights,
                  d_decode_swiglu_int8_,
                  d_decode_swiglu_scales_,
                  d_output,
                  top_k,
                  d_model,
                  intermediate,
                  table.num_experts,
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
